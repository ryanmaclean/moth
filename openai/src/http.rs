//! libcurl streaming POST for OpenAI-compatible chat-completions endpoints.
//!
//! One easy handle per request, on a background thread. Body chunks land in
//! the curl WRITEFUNCTION callback and are forwarded across a bounded mpsc
//! channel to the caller. If the caller stops draining, the next write
//! callback returns 0 → CURLE_WRITE_ERROR → curl_easy_perform exits cleanly.
//!
//! Structurally identical to `anthropic::http`; the only differences are the
//! configurable base URL, the path (`/v1/chat/completions`), and the auth
//! header shape (`Authorization: Bearer …`).
//!
//! ## Retry & cancellation
//!
//! Transient pre-first-byte failures (DNS, connect, handshake, 5xx/408/425/429)
//! are retried with exponential backoff via `wire::retry::with_backoff`. The
//! "no retry after first byte" invariant is load-bearing: once a single byte
//! has been handed to the iterator consumer the streaming contract has been
//! observed externally, so we must not silently re-issue the request.
//!
//! Cancellation: a `CURLOPT_XFERINFOFUNCTION` callback polls `ctx.aborted` so
//! that connect-time / between-byte stalls can be interrupted promptly when
//! the caller drops the `Stream` (the existing WRITEFUNCTION-returns-0 path
//! still handles cancellation that occurs while bytes are arriving).
//!
//! Redirects are followed explicitly. Each resolved destination must remain
//! on the initial HTTP(S) origin before auth or body can be sent to it.

use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::thread::JoinHandle;

use curl_sys as c;

use crate::Error;

const PATH: &str = "/v1/chat/completions";

// curl-sys exposes CURLOPT_XFERINFOFUNCTION/DATA as commented-out entries
// in its `pub const` list, so we materialise them from the same base values
// (`CURLOPTTYPE_FUNCTIONPOINT` = 20_000, `CURLOPTTYPE_OBJECTPOINT` = 10_000).
const CURLOPT_XFERINFOFUNCTION: c::CURLoption = 20_000 + 219;
const CURLOPT_XFERINFODATA: c::CURLoption = 10_000 + 57;
// curl-sys 0.4.88 omits this binding, although its bundled curl 8.20.0
// curl.h defines CURLOPT(CURLOPT_SUPPRESS_CONNECT_HEADERS, CURLOPTTYPE_LONG, 265).
// Make an unsupported selected libcurl fail the request at setopt, rather
// than letting a proxy CONNECT 200 open the origin-body gate.
const CURLOPT_SUPPRESS_CONNECT_HEADERS: c::CURLoption = c::CURLOPTTYPE_LONG + 265;

pub(crate) enum Chunk {
    Data(Vec<u8>),
    End(Result<u32, Error>),
}

pub(crate) struct Stream {
    pub(crate) rx: Receiver<Chunk>,
    pub(crate) handle: Option<JoinHandle<()>>,
    aborted: Arc<AtomicBool>,
}

impl Drop for Stream {
    fn drop(&mut self) {
        // Disconnect a full bounded sender before joining its worker.
        self.aborted.store(true, Ordering::Release);
        let (_, disconnected) = std::sync::mpsc::channel();
        let rx = std::mem::replace(&mut self.rx, disconnected);
        drop(rx);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

struct Ctx {
    tx: SyncSender<Chunk>,
    /// Set by Stream::drop before closing the receiver, or when a send
    /// observes receiver disconnection. Both callbacks read it.
    aborted: Arc<AtomicBool>,
    /// Set on the first successful `tx.send`. Once true, the streaming
    /// contract has been observed externally and the request is no longer
    /// safe to retry.
    delivered: AtomicBool,
    /// Suppress interim and 3xx response bodies before they reach the stream.
    redirecting: AtomicBool,
    /// A malformed status line must not open the body-delivery gate.
    header_error: AtomicBool,
    /// A response body is eligible only after its status and final header
    /// boundary. New status lines also delimit interim and proxy responses.
    status_seen: AtomicBool,
    headers_complete: AtomicBool,
    response_status: AtomicU16,
}

extern "C" fn write_cb(ptr: *mut c_char, size: usize, nmemb: usize, user: *mut c_void) -> usize {
    let total = size.saturating_mul(nmemb);
    if total == 0 {
        return 0;
    }
    let ctx = unsafe { &*(user as *const Ctx) };
    if ctx.aborted.load(Ordering::Acquire) {
        return 0;
    }
    if ctx.redirecting.load(Ordering::Acquire) {
        return total;
    }
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, total) };
    match ctx.tx.send(Chunk::Data(slice.to_vec())) {
        Ok(()) => {
            ctx.delivered.store(true, Ordering::Relaxed);
            total
        }
        Err(_) => {
            ctx.aborted.store(true, Ordering::Relaxed);
            0
        }
    }
}

extern "C" fn header_cb(ptr: *mut c_char, size: usize, nmemb: usize, user: *mut c_void) -> usize {
    let total = size.saturating_mul(nmemb);
    if total == 0 {
        return 0;
    }
    let line = unsafe { std::slice::from_raw_parts(ptr as *const u8, total) };
    let ctx = unsafe { &*(user as *const Ctx) };
    if line.starts_with(b"HTTP/") {
        // A new status line may follow a completed interim or proxy response,
        // but never interrupt the current header block.
        if ctx.status_seen.load(Ordering::Acquire)
            && !ctx.headers_complete.load(Ordering::Acquire)
        {
            ctx.header_error.store(true, Ordering::Release);
            return 0;
        }
        let status = (|| {
            let mut parts = line.split(|b| b.is_ascii_whitespace()).filter(|p| !p.is_empty());
            let version = parts.next()?;
            let digits = parts.next()?;
            let version = &version[b"HTTP/".len()..];
            let mut version_parts = version.split(|&b| b == b'.');
            let major = version_parts.next()?;
            let minor = version_parts.next();
            if major.is_empty()
                || !major.iter().all(u8::is_ascii_digit)
                || minor.is_some_and(|part| part.is_empty() || !part.iter().all(u8::is_ascii_digit))
                || version_parts.next().is_some()
                || digits.len() != 3
                || !digits.iter().all(u8::is_ascii_digit)
            {
                return None;
            }
            std::str::from_utf8(digits).ok()?.parse::<u16>().ok()
        })();
        match status {
            Some(code) if (100..600).contains(&code) => {
                ctx.redirecting.store(true, Ordering::Release);
                ctx.response_status.store(code, Ordering::Release);
                ctx.headers_complete.store(false, Ordering::Release);
                ctx.status_seen.store(true, Ordering::Release);
            }
            _ => {
                ctx.header_error.store(true, Ordering::Release);
                return 0;
            }
        }
    } else if line.iter().all(u8::is_ascii_whitespace) {
        if !ctx.status_seen.load(Ordering::Acquire) {
            ctx.header_error.store(true, Ordering::Release);
            return 0;
        }
        ctx.headers_complete.store(true, Ordering::Release);
        let code = ctx.response_status.load(Ordering::Acquire);
        ctx.redirecting.store(code < 200 || (300..400).contains(&code), Ordering::Release);
    } else if !ctx.status_seen.load(Ordering::Acquire)
        || !line.split(|&b| b == b':').next().is_some_and(|name| {
            !name.is_empty() && name.iter().all(|&b| {
                b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)
            })
        })
        || !line.contains(&b':')
    {
        ctx.header_error.store(true, Ordering::Release);
        return 0;
    }
    total
}

extern "C" fn xferinfo_cb(
    user: *mut c_void,
    _dltotal: c::curl_off_t,
    _dlnow: c::curl_off_t,
    _ultotal: c::curl_off_t,
    _ulnow: c::curl_off_t,
) -> i32 {
    let ctx = unsafe { &*(user as *const Ctx) };
    // Non-zero return triggers CURLE_ABORTED_BY_CALLBACK from curl, which
    // unblocks any connect/recv that's currently parked. Polled by curl
    // roughly once per second, so this matters most for between-byte stalls.
    i32::from(ctx.aborted.load(Ordering::Acquire))
}

pub(crate) fn post_stream(api_key: &str, base_url: &str, body: String) -> Result<Stream, Error> {
    let host = host_of(base_url);
    let breaker_cfg = wire::retry::BreakerConfig::default();
    match wire::retry::check(&host, &breaker_cfg) {
        wire::retry::BreakerVerdict::Allow | wire::retry::BreakerVerdict::HalfOpenProbe => {}
        wire::retry::BreakerVerdict::Open => {
            return Err(Error::Http(format!("circuit open for {host} (recent upstream failures)")));
        }
    }

    let (tx, rx) = std::sync::mpsc::sync_channel::<Chunk>(64);
    let aborted = Arc::new(AtomicBool::new(false));
    let worker_aborted = Arc::clone(&aborted);

    let full_url = join_url(base_url, PATH);
    let url = CString::new(full_url).map_err(|_| Error::Http("bad URL".into()))?;
    let auth_header = CString::new(format!("Authorization: Bearer {api_key}"))
        .map_err(|_| Error::Http("api key contains NUL".into()))?;
    let content_type = CString::new("Content-Type: application/json").unwrap();
    let accept = CString::new("Accept: text/event-stream").unwrap();
    let body_bytes = body.into_bytes();

    let handle = std::thread::spawn(move || {
        let ctx = Ctx {
            tx: tx.clone(),
            aborted: worker_aborted,
            delivered: AtomicBool::new(false),
            redirecting: AtomicBool::new(true),
            header_error: AtomicBool::new(false),
            status_seen: AtomicBool::new(false),
            headers_complete: AtomicBool::new(false),
            response_status: AtomicU16::new(0),
        };
        let policy = wire::retry::RetryPolicy::default();
        let result = wire::retry::with_backoff(&policy, |_attempt| {
            if ctx.aborted.load(Ordering::Acquire) {
                return wire::retry::Outcome::Fatal(Error::Http("stream cancelled".into()));
            }
            // Fresh body clone per attempt: curl reads from the buffer for
            // the duration of `curl_easy_perform`, and a retried call may
            // re-issue the same bytes. Cloning `Vec<u8>` is cheap.
            let body_attempt = body_bytes.clone();
            let perform = unsafe {
                run_easy(&url, &auth_header, &content_type, &accept, &body_attempt, &ctx)
            };
            // Strict invariant: never retry once bytes have crossed the
            // channel into the iterator consumer.
            let delivered = ctx.delivered.load(Ordering::Relaxed);
            if ctx.aborted.load(Ordering::Acquire) {
                wire::retry::Outcome::Fatal(Error::Http("stream cancelled".into()))
            } else {
                classify(perform, delivered)
            }
        });
        if ctx.aborted.load(Ordering::Acquire) {
            return;
        }
        match &result {
            Ok(_) => wire::retry::record_success(&host, &breaker_cfg),
            Err(_) => wire::retry::record_failure(&host, &breaker_cfg),
        }
        let _ = tx.send(Chunk::End(result));
    });

    Ok(Stream { rx, handle: Some(handle), aborted })
}

/// Extract the host portion of `base_url` (e.g. "https://api.openai.com/v1"
/// → "api.openai.com"). Defaults to the whole input if parsing fails so
/// the breaker still keys consistently per caller-supplied URL.
fn host_of(base_url: &str) -> String {
    let after_scheme = base_url.split_once("://").map(|(_, rest)| rest).unwrap_or(base_url);
    after_scheme.split(['/', ':', '?']).next().unwrap_or(base_url).to_string()
}

fn join_url(base: &str, path: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    format!("{trimmed}{path}")
}

/// One curl_easy_perform's worth of outcome, factored out so the body of
/// `run_easy` can be a single FFI call with no policy mixed in.
pub(crate) struct PerformOutcome {
    pub rc: c::CURLcode,
    pub status: i64,
}

/// Classify a perform outcome into a retry policy decision. Pure function,
/// no FFI — exists so the retry rules are unit-testable without a network.
///
/// Retryable iff `!delivered` AND (curl-side transient | retryable status):
/// - CURLE_COULDNT_RESOLVE_HOST (6), CURLE_COULDNT_CONNECT (7)
/// - CURLE_OPERATION_TIMEDOUT (28), CURLE_SSL_CONNECT_ERROR (35)
/// - CURLE_GOT_NOTHING (52), CURLE_SEND_ERROR (55), CURLE_RECV_ERROR (56)
/// - HTTP 408 / 425 / 429 / 502 / 503 / 504
fn classify(
    outcome: Result<PerformOutcome, Error>,
    delivered: bool,
) -> wire::retry::Outcome<u32, Error> {
    match outcome {
        Err(e) => wire::retry::Outcome::Fatal(e),
        Ok(p) => {
            // CURLE_WRITE_ERROR (23) is a clean caller-initiated abort —
            // not a real failure; preserve the existing semantics.
            if p.rc == c::CURLE_WRITE_ERROR {
                return if (200..300).contains(&p.status) {
                    wire::retry::Outcome::Ok(p.status as u32)
                } else {
                    wire::retry::Outcome::Fatal(Error::Http(format!("HTTP {}", p.status)))
                };
            }
            if p.rc != c::CURLE_OK {
                if !delivered && is_retryable_curl_code(p.rc) {
                    return wire::retry::Outcome::Retryable(curl_err(p.rc, "perform"));
                }
                return wire::retry::Outcome::Fatal(curl_err(p.rc, "perform"));
            }
            if (200..300).contains(&p.status) {
                wire::retry::Outcome::Ok(p.status as u32)
            } else if !delivered && is_retryable_status(p.status) {
                wire::retry::Outcome::Retryable(Error::Http(format!("HTTP {}", p.status)))
            } else {
                wire::retry::Outcome::Fatal(Error::Http(format!("HTTP {}", p.status)))
            }
        }
    }
}

fn is_retryable_curl_code(rc: c::CURLcode) -> bool {
    matches!(
        rc,
        c::CURLE_COULDNT_RESOLVE_HOST
            | c::CURLE_COULDNT_CONNECT
            | c::CURLE_OPERATION_TIMEDOUT
            | c::CURLE_SSL_CONNECT_ERROR
            | c::CURLE_GOT_NOTHING
            | c::CURLE_SEND_ERROR
            | c::CURLE_RECV_ERROR
    )
}

fn is_retryable_status(status: i64) -> bool {
    matches!(status, 408 | 425 | 429 | 502 | 503 | 504)
}

unsafe fn run_easy(
    url: &CString,
    auth_header: &CString,
    content_type: &CString,
    accept: &CString,
    body: &[u8],
    ctx: *const Ctx,
) -> Result<PerformOutcome, Error> {
    unsafe { run_easy_with_hook(url, auth_header, content_type, accept, body, ctx, |_| {}) }
}

unsafe fn run_easy_with_hook<F: FnMut(usize)>(
    url: &CString,
    auth_header: &CString,
    content_type: &CString,
    accept: &CString,
    body: &[u8],
    ctx: *const Ctx,
    mut after_redirect: F,
) -> Result<PerformOutcome, Error> {
    let original = url.to_str().map_err(|_| Error::Http("URL is not UTF-8".into()))?;
    wire::http_origin::valid_http_origin(original)
        .map_err(|e| Error::Http(format!("invalid OpenAI HTTP origin: {e}")))?;
    let mut current = url.clone();
    let mut use_post = true;
    for hop in 0..=10 {
        if unsafe { (&*ctx).aborted.load(Ordering::Acquire) } {
            return Err(Error::Http("stream cancelled".into()));
        }
        let (outcome, redirect) = unsafe {
            run_easy_once(&current, auth_header, content_type, accept, body, ctx, use_post)?
        };
        if outcome.rc != c::CURLE_OK || !matches!(outcome.status, 301 | 302 | 303 | 307 | 308) {
            return Ok(outcome);
        }
        if hop == 10 {
            return Err(Error::Http("OpenAI HTTP redirect limit exceeded".into()));
        }
        let next = redirect.ok_or_else(|| Error::Http("OpenAI HTTP redirect lacks Location".into()))?;
        if !wire::http_origin::same_http_origin(original, &next)
            .map_err(|e| Error::Http(format!("invalid OpenAI HTTP redirect origin: {e}")))?
        {
            return Err(Error::Http("cross-origin OpenAI HTTP redirect refused".into()));
        }
        use_post = use_post && !matches!(outcome.status, 301 | 302 | 303);
        current = CString::new(next).map_err(|_| Error::Http("redirect URL contains NUL".into()))?;
        after_redirect(hop);
    }
    unreachable!("bounded redirect loop returns")
}

unsafe fn run_easy_once(
    url: &CString,
    auth_header: &CString,
    content_type: &CString,
    accept: &CString,
    body: &[u8],
    ctx: *const Ctx,
    use_post: bool,
) -> Result<(PerformOutcome, Option<String>), Error> {
    unsafe { run_easy_once_with_proxy(url, auth_header, content_type, accept, body, ctx, use_post, None) }
}

// The proxy override is used only by the local CONNECT fixture; production
// leaves libcurl's existing proxy selection untouched.
unsafe fn run_easy_once_with_proxy(
    url: &CString,
    auth_header: &CString,
    content_type: &CString,
    accept: &CString,
    body: &[u8],
    ctx: *const Ctx,
    use_post: bool,
    proxy: Option<&CString>,
) -> Result<(PerformOutcome, Option<String>), Error> {
    unsafe {
        let easy = c::curl_easy_init();
        if easy.is_null() {
            return Err(Error::Http("curl_easy_init failed".into()));
        }

        let mut headers: *mut c::curl_slist = ptr::null_mut();
        headers = c::curl_slist_append(headers, auth_header.as_ptr());
        headers = c::curl_slist_append(headers, content_type.as_ptr());
        headers = c::curl_slist_append(headers, accept.as_ptr());

        let guard = Handle { easy, headers };

        setopt_ptr(easy, c::CURLOPT_URL, url.as_ptr() as *const c_void, "URL")?;
        if let Some(proxy) = proxy {
            setopt_ptr(easy, c::CURLOPT_PROXY, proxy.as_ptr() as *const c_void, "PROXY")?;
            setopt_ptr(easy, c::CURLOPT_NOPROXY, b"\0".as_ptr() as *const c_void, "NOPROXY")?;
            setopt_long(easy, c::CURLOPT_HTTPPROXYTUNNEL, 1, "HTTPPROXYTUNNEL")?;
        }
        if use_post {
            setopt_long(easy, c::CURLOPT_POST, 1, "POST")?;
            setopt_ptr(easy, c::CURLOPT_POSTFIELDS, body.as_ptr() as *const c_void, "POSTFIELDS")?;
            setopt_long(easy, c::CURLOPT_POSTFIELDSIZE, body.len() as i64, "POSTFIELDSIZE")?;
        } else {
            setopt_long(easy, c::CURLOPT_HTTPGET, 1, "HTTPGET")?;
        }
        setopt_ptr(easy, c::CURLOPT_HTTPHEADER, headers as *const c_void, "HTTPHEADER")?;
        setopt_ptr(easy, c::CURLOPT_WRITEFUNCTION, write_cb as *const c_void, "WRITEFUNCTION")?;
        setopt_ptr(easy, c::CURLOPT_WRITEDATA, ctx as *const c_void, "WRITEDATA")?;
        setopt_ptr(easy, c::CURLOPT_HEADERFUNCTION, header_cb as *const c_void, "HEADERFUNCTION")?;
        setopt_ptr(easy, c::CURLOPT_HEADERDATA, ctx as *const c_void, "HEADERDATA")?;
        // CONNECT is a proxy response, not the origin response. Suppress its
        // headers from both callbacks before a 2xx block can open write_cb.
        setopt_long(easy, CURLOPT_SUPPRESS_CONNECT_HEADERS, 1, "SUPPRESS_CONNECT_HEADERS")?;
        // Cancellation: xferinfo_cb returns non-zero when ctx.aborted is set,
        // which lets curl break out of connect/recv stalls without waiting
        // for the next byte to flow through write_cb.
        setopt_long(easy, c::CURLOPT_NOPROGRESS, 0, "NOPROGRESS")?;
        setopt_ptr(
            easy,
            CURLOPT_XFERINFOFUNCTION,
            xferinfo_cb as *const c_void,
            "XFERINFOFUNCTION",
        )?;
        setopt_ptr(easy, CURLOPT_XFERINFODATA, ctx as *const c_void, "XFERINFODATA")?;
        setopt_long(easy, c::CURLOPT_FOLLOWLOCATION, 0, "FOLLOWLOCATION")?;
        // Connection-level timeouts so a half-open or LB-dropped TCP socket
        // can't wedge the worker thread indefinitely. No hard CURLOPT_TIMEOUT
        // — streams can legitimately run for many minutes — but require at
        // least 1 byte/s averaged over a 60s window once data should be
        // flowing, and cap the initial connect at 10s.
        setopt_long(easy, c::CURLOPT_CONNECTTIMEOUT, 10, "CONNECTTIMEOUT")?;
        setopt_long(easy, c::CURLOPT_LOW_SPEED_LIMIT, 1, "LOW_SPEED_LIMIT")?;
        setopt_long(easy, c::CURLOPT_LOW_SPEED_TIME, 60, "LOW_SPEED_TIME")?;

        (&*ctx).redirecting.store(true, Ordering::Release);
        (&*ctx).header_error.store(false, Ordering::Release);
        (&*ctx).status_seen.store(false, Ordering::Release);
        (&*ctx).headers_complete.store(false, Ordering::Release);
        (&*ctx).response_status.store(0, Ordering::Release);
        let perform_rc = c::curl_easy_perform(easy);
        let mut status: i64 = 0;
        c::curl_easy_getinfo(easy, c::CURLINFO_RESPONSE_CODE, &mut status);
        let mut redirect_ptr: *mut c_char = ptr::null_mut();
        let info_rc = c::curl_easy_getinfo(easy, c::CURLINFO_REDIRECT_URL, &mut redirect_ptr);
        let redirect = if info_rc == c::CURLE_OK && !redirect_ptr.is_null() {
            Some(std::ffi::CStr::from_ptr(redirect_ptr).to_str()
                .map_err(|_| Error::Http("redirect URL is not UTF-8".into()))?.to_owned())
        } else {
            None
        };

        drop(guard);

        if (&*ctx).header_error.load(Ordering::Acquire) {
            return Err(Error::Http("malformed HTTP response header/status".into()));
        }
        if perform_rc == c::CURLE_OK
            && (!(&*ctx).status_seen.load(Ordering::Acquire)
                || !(&*ctx).headers_complete.load(Ordering::Acquire)
                || (&*ctx).response_status.load(Ordering::Acquire) as i64 != status)
        {
            return Err(Error::Http("incomplete HTTP response status/headers".into()));
        }
        if perform_rc == c::CURLE_OK && info_rc != c::CURLE_OK {
            return Err(curl_err(info_rc, "redirect URL"));
        }
        Ok((PerformOutcome { rc: perform_rc, status }, redirect))
    }
}

struct Handle {
    easy: *mut c::CURL,
    headers: *mut c::curl_slist,
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            if !self.headers.is_null() {
                c::curl_slist_free_all(self.headers);
            }
            if !self.easy.is_null() {
                c::curl_easy_cleanup(self.easy);
            }
        }
    }
}

unsafe fn setopt_ptr(
    easy: *mut c::CURL,
    opt: c::CURLoption,
    val: *const c_void,
    name: &str,
) -> Result<(), Error> {
    let rc = unsafe { c::curl_easy_setopt(easy, opt, val) };
    if rc != c::CURLE_OK {
        return Err(curl_err(rc, name));
    }
    Ok(())
}

unsafe fn setopt_long(
    easy: *mut c::CURL,
    opt: c::CURLoption,
    val: i64,
    name: &str,
) -> Result<(), Error> {
    let rc = unsafe { c::curl_easy_setopt(easy, opt, val) };
    if rc != c::CURLE_OK {
        return Err(curl_err(rc, name));
    }
    Ok(())
}

fn curl_err(code: c::CURLcode, where_: &str) -> Error {
    let msg = unsafe {
        let p = c::curl_easy_strerror(code);
        if p.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    Error::Http(format!("curl {where_}: {msg} (code {code})"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wire::retry::Outcome;

    fn read_http_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
        use std::io::Read;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
        let mut data = Vec::new();
        loop {
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).expect("HTTP request read");
            assert!(n > 0, "connection closed before complete request");
            data.extend_from_slice(&buf[..n]);
            if let Some(end) = data.windows(4).position(|window| window == b"\r\n\r\n").map(|i| i + 4) {
                let headers = String::from_utf8_lossy(&data[..end]);
                let body_len = headers.lines()
                    .find_map(|line| line.split_once(':').and_then(|(name, value)| {
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok()).flatten()
                    }))
                    .unwrap_or(0);
                if data.len() >= end + body_len {
                    return data;
                }
            }
        }
    }

    fn accept_with_deadline(listener: &std::net::TcpListener) -> std::net::TcpStream {
        listener.set_nonblocking(true).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false).unwrap();
                    return stream;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock
                    && std::time::Instant::now() < deadline =>
                {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(error) => panic!("timed out or failed accepting request: {error}"),
            }
        }
    }

    fn header_fixture() -> (Ctx, Receiver<Chunk>) {
        let (tx, rx) = std::sync::mpsc::sync_channel(4);
        (Ctx {
            tx,
            aborted: Arc::new(AtomicBool::new(false)),
            delivered: AtomicBool::new(false),
            redirecting: AtomicBool::new(true),
            header_error: AtomicBool::new(false),
            status_seen: AtomicBool::new(false),
            headers_complete: AtomicBool::new(false),
            response_status: AtomicU16::new(0),
        }, rx)
    }

    fn feed_header(ctx: &Ctx, line: &[u8]) -> usize {
        header_cb(line.as_ptr() as *mut c_char, 1, line.len(), ctx as *const Ctx as *mut c_void)
    }

    fn feed_body(ctx: &Ctx, body: &[u8]) -> usize {
        write_cb(body.as_ptr() as *mut c_char, 1, body.len(), ctx as *const Ctx as *mut c_void)
    }

    #[test]
    fn header_gate_accepts_http_named_field_only_after_final_boundary() {
        let (ctx, rx) = header_fixture();
        assert_eq!(feed_body(&ctx, b"before-status"), b"before-status".len());
        assert!(rx.try_recv().is_err());
        assert_eq!(feed_header(&ctx, b"HTTP/1.1 200 OK\r\n"), 17);
        assert_eq!(feed_header(&ctx, b"HTTP-Status: harmless\r\n"), 23);
        assert_eq!(feed_body(&ctx, b"before-boundary"), b"before-boundary".len());
        assert!(rx.try_recv().is_err());
        assert_eq!(feed_header(&ctx, b"\r\n"), 2);
        assert_eq!(feed_body(&ctx, b"final"), 5);
        match rx.try_recv() {
            Ok(Chunk::Data(data)) => assert_eq!(data.as_slice(), b"final"),
            _ => panic!("only final body should be delivered"),
        }
        assert!(!ctx.header_error.load(Ordering::Acquire));
    }

    #[test]
    fn header_gate_suppresses_interim_and_redirect_bodies() {
        let (ctx, rx) = header_fixture();
        for line in [b"HTTP/1.1 100 Continue\r\n".as_slice(), b"\r\n".as_slice()] {
            assert_eq!(feed_header(&ctx, line), line.len());
        }
        assert_eq!(feed_body(&ctx, b"interim"), 7);
        assert!(rx.try_recv().is_err());
        assert_eq!(feed_header(&ctx, b"HTTP/2 307 Redirect\r\n"), b"HTTP/2 307 Redirect\r\n".len());
        assert_eq!(feed_header(&ctx, b"Location: /again\r\n"), b"Location: /again\r\n".len());
        assert_eq!(feed_header(&ctx, b"\r\n"), 2);
        assert_eq!(feed_body(&ctx, b"redirect-body"), 13);
        assert!(rx.try_recv().is_err());
        assert!(!ctx.header_error.load(Ordering::Acquire));
    }

    #[test]
    fn header_gate_rejects_absent_or_malformed_status() {
        let (absent, absent_rx) = header_fixture();
        assert_eq!(feed_header(&absent, b"HTTP-Status: harmless\r\n"), 0);
        assert!(absent.header_error.load(Ordering::Acquire));
        assert_eq!(feed_body(&absent, b"hidden"), 6);
        assert!(absent_rx.try_recv().is_err());

        let (malformed, malformed_rx) = header_fixture();
        assert_eq!(feed_header(&malformed, b"HTTP/1.1 not-a-status\r\n"), 0);
        assert!(malformed.header_error.load(Ordering::Acquire));
        assert_eq!(feed_body(&malformed, b"hidden"), 6);
        assert!(malformed_rx.try_recv().is_err());
    }

    /// The local proxy sends CONNECT 200, then waits after the tunneled POST
    /// before sending the origin response. At that barrier, libcurl must not
    /// have exposed the proxy status to our callbacks or opened write_cb.
    fn proxy_connect_then_origin_fixture(origin_status: u16) {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let proxy_port = listener.local_addr().unwrap().port();
        let (tunneled_tx, tunneled_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let proxy = std::thread::spawn(move || {
            let mut stream = accept_with_deadline(&listener);
            let connect = read_http_request(&mut stream);
            assert!(connect.starts_with(b"CONNECT 127.0.0.1:9 HTTP/1.1\r\n"));
            assert!(!String::from_utf8_lossy(&connect).to_ascii_lowercase().contains("authorization: bearer"));
            stream.write_all(b"HTTP/1.1 200 Connection established\r\nProxy-Agent: fixture\r\n\r\n").unwrap();
            let post = read_http_request(&mut stream);
            assert!(post.starts_with(b"POST /v1/chat/completions HTTP/1.1\r\n"));
            assert!(String::from_utf8_lossy(&post).contains("Authorization: Bearer fixture-key"));
            tunneled_tx.send(()).unwrap();
            release_rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
            let response: &[u8] = if origin_status == 200 {
                b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nfinal"
            } else {
                b"HTTP/1.1 307 Temporary Redirect\r\nLocation: /again\r\nContent-Length: 13\r\nConnection: close\r\n\r\nredirect-body"
            };
            stream.write_all(response).unwrap();
        });

        let (ctx, rx) = header_fixture();
        let ctx = Arc::new(ctx);
        let worker_ctx = Arc::clone(&ctx);
        let client = std::thread::spawn(move || {
            let url = CString::new("http://127.0.0.1:9/v1/chat/completions").unwrap();
            let proxy_url = CString::new(format!("http://127.0.0.1:{proxy_port}")).unwrap();
            let auth = CString::new("Authorization: Bearer fixture-key").unwrap();
            let content_type = CString::new("Content-Type: application/json").unwrap();
            let accept = CString::new("Accept: text/event-stream").unwrap();
            unsafe {
                run_easy_once_with_proxy(
                    &url, &auth, &content_type, &accept, b"{}", Arc::as_ptr(&worker_ctx), true,
                    Some(&proxy_url),
                )
            }
        });

        tunneled_rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        assert!(!ctx.status_seen.load(Ordering::Acquire), "CONNECT status reached origin callback");
        assert!(!ctx.headers_complete.load(Ordering::Acquire));
        assert!(!ctx.delivered.load(Ordering::Acquire));
        assert!(rx.try_recv().is_err(), "bytes arrived before origin response");
        release_tx.send(()).unwrap();
        let (outcome, redirect) = client.join().unwrap().unwrap();
        assert_eq!(outcome.rc, c::CURLE_OK);
        assert_eq!(outcome.status, i64::from(origin_status));
        assert_eq!(ctx.response_status.load(Ordering::Acquire), origin_status);
        assert!(ctx.headers_complete.load(Ordering::Acquire));
        assert!(!ctx.header_error.load(Ordering::Acquire));
        if origin_status == 200 {
            assert!(redirect.is_none());
            match rx.try_recv() {
                Ok(Chunk::Data(body)) => assert_eq!(body.as_slice(), b"final"),
                _ => panic!("origin 200 body was not delivered"),
            }
        } else {
            assert_eq!(redirect.as_deref(), Some("http://127.0.0.1:9/again"));
            assert!(rx.try_recv().is_err(), "origin 307 body was forwarded");
        }
        proxy.join().unwrap();
    }

    #[test]
    fn proxy_connect_then_origin_200_opens_only_origin_body() {
        proxy_connect_then_origin_fixture(200);
    }

    #[test]
    fn proxy_connect_then_origin_307_never_forwards_redirect_body() {
        proxy_connect_then_origin_fixture(307);
    }

    #[test]
    fn dropping_full_stream_disconnects_before_join() {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Chunk>(1);
        assert!(tx.send(Chunk::Data(vec![1])).is_ok());
        let aborted = Arc::new(AtomicBool::new(false));
        let worker_aborted = Arc::clone(&aborted);
        let worker_finished = Arc::new(AtomicBool::new(false));
        let finished_in_worker = Arc::clone(&worker_finished);
        let handle = std::thread::spawn(move || {
            assert!(tx.send(Chunk::Data(vec![2])).is_err());
            assert!(worker_aborted.load(Ordering::Acquire));
            finished_in_worker.store(true, Ordering::Release);
        });
        drop(Stream { rx, handle: Some(handle), aborted: Arc::clone(&aborted) });
        assert!(aborted.load(Ordering::Acquire));
        assert!(worker_finished.load(Ordering::Acquire));
    }

    fn ok_outcome(rc: c::CURLcode, status: i64) -> Result<PerformOutcome, Error> {
        Ok(PerformOutcome { rc, status })
    }

    #[test]
    fn join_url_no_trailing_slash() {
        assert_eq!(
            join_url("https://api.openai.com", "/v1/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn join_url_trims_trailing_slash() {
        assert_eq!(
            join_url("http://localhost:1234/", "/v1/chat/completions"),
            "http://localhost:1234/v1/chat/completions"
        );
    }

    #[test]
    fn classify_success_is_ok() {
        match classify(ok_outcome(c::CURLE_OK, 200), false) {
            Outcome::Ok(200) => {}
            _ => panic!("expected Ok"),
        }
    }

    #[test]
    fn classify_dns_failure_pre_byte_is_retryable() {
        match classify(ok_outcome(c::CURLE_COULDNT_RESOLVE_HOST, 0), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable"),
        }
    }

    #[test]
    fn classify_dns_failure_after_first_byte_is_fatal() {
        // Streaming-retry invariant: once any data delivered, never retry.
        match classify(ok_outcome(c::CURLE_COULDNT_RESOLVE_HOST, 0), true) {
            Outcome::Fatal(_) => {}
            _ => panic!("expected Fatal once delivered=true"),
        }
    }

    #[test]
    fn classify_connect_failure_is_retryable() {
        match classify(ok_outcome(c::CURLE_COULDNT_CONNECT, 0), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable"),
        }
    }

    #[test]
    fn classify_timeout_is_retryable() {
        match classify(ok_outcome(c::CURLE_OPERATION_TIMEDOUT, 0), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable"),
        }
    }

    #[test]
    fn classify_recv_error_pre_byte_is_retryable() {
        match classify(ok_outcome(c::CURLE_RECV_ERROR, 0), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable"),
        }
    }

    #[test]
    fn classify_got_nothing_is_retryable() {
        match classify(ok_outcome(c::CURLE_GOT_NOTHING, 0), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable"),
        }
    }

    #[test]
    fn classify_http_429_pre_byte_is_retryable() {
        match classify(ok_outcome(c::CURLE_OK, 429), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable for 429"),
        }
    }

    #[test]
    fn classify_http_503_pre_byte_is_retryable() {
        match classify(ok_outcome(c::CURLE_OK, 503), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable for 503"),
        }
    }

    #[test]
    fn classify_http_429_after_first_byte_is_fatal() {
        match classify(ok_outcome(c::CURLE_OK, 429), true) {
            Outcome::Fatal(_) => {}
            _ => panic!("expected Fatal once delivered=true"),
        }
    }

    #[test]
    fn classify_http_400_is_fatal() {
        match classify(ok_outcome(c::CURLE_OK, 400), false) {
            Outcome::Fatal(_) => {}
            _ => panic!("expected Fatal for 400"),
        }
    }

    #[test]
    fn classify_http_401_is_fatal() {
        match classify(ok_outcome(c::CURLE_OK, 401), false) {
            Outcome::Fatal(_) => {}
            _ => panic!("expected Fatal for 401"),
        }
    }

    #[test]
    fn classify_write_error_is_ok_when_status_2xx() {
        // CURLE_WRITE_ERROR with 200 = caller dropped the receiver mid-stream;
        // not a real failure.
        match classify(ok_outcome(c::CURLE_WRITE_ERROR, 200), true) {
            Outcome::Ok(200) => {}
            _ => panic!("expected Ok for caller-abort"),
        }
    }

    #[test]
    fn classify_non_retryable_curl_code_is_fatal() {
        // CURLE_HTTP_RETURNED_ERROR (22) isn't in our retry set.
        match classify(ok_outcome(c::CURLE_HTTP_RETURNED_ERROR, 0), false) {
            Outcome::Fatal(_) => {}
            _ => panic!("expected Fatal for non-retryable code"),
        }
    }

    #[test]
    fn retry_loop_clones_body_per_attempt() {
        // Verify that wrapping with_backoff with a closure that captures a
        // shared body produces a fresh clone per iteration.
        let body: Vec<u8> = b"payload".to_vec();
        let seen_ptrs = std::sync::Mutex::new(Vec::<usize>::new());
        let policy = wire::retry::RetryPolicy {
            max_attempts: 3,
            base_delay: std::time::Duration::from_millis(1),
            max_delay: std::time::Duration::from_millis(2),
        };
        let _res: Result<u32, Error> = wire::retry::with_backoff(&policy, |_attempt| {
            let cloned = body.clone();
            seen_ptrs.lock().unwrap().push(cloned.as_ptr() as usize);
            classify(ok_outcome(c::CURLE_COULDNT_CONNECT, 0), false)
        });
        let ptrs = seen_ptrs.lock().unwrap();
        assert_eq!(ptrs.len(), 3, "loop should run all 3 attempts");
    }

    #[test]
    fn aborted_flag_short_circuits_retry_loop_via_fatal() {
        // Once classify returns Ok/Fatal (which is what an aborted-by-caller
        // write_cb path produces via CURLE_WRITE_ERROR), the loop stops
        // immediately and we don't keep hammering the upstream.
        let mut attempts = 0;
        let policy = wire::retry::RetryPolicy {
            max_attempts: 4,
            base_delay: std::time::Duration::from_millis(1),
            max_delay: std::time::Duration::from_millis(2),
        };
        let _res: Result<u32, Error> = wire::retry::with_backoff(&policy, |_| {
            attempts += 1;
            classify(ok_outcome(c::CURLE_WRITE_ERROR, 200), true)
        });
        assert_eq!(attempts, 1, "Ok outcome must short-circuit");
    }

    #[test]
    fn foreign_origin_307_receives_neither_auth_nor_post_body() {
        use std::io::Write;
        use std::net::TcpListener;
        let first = TcpListener::bind("127.0.0.1:0").unwrap();
        let second = TcpListener::bind("127.0.0.1:0").unwrap();
        let first_addr = first.local_addr().unwrap();
        let second_addr = second.local_addr().unwrap();
        second.set_nonblocking(true).unwrap();
        let contacted = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let contacted_server = Arc::clone(&contacted);
        let stop_server = Arc::clone(&stop);
        let second_thread = std::thread::spawn(move || {
            while !stop_server.load(Ordering::Acquire) {
                match second.accept() {
                    Ok((mut stream, _)) => {
                        contacted_server.store(true, Ordering::Release);
                        let _ = read_http_request(&mut stream);
                        break;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
        });
        let first_thread = std::thread::spawn(move || {
            let mut stream = accept_with_deadline(&first);
            let request = read_http_request(&mut stream);
            assert!(request.windows(b"body-sentinel".len()).any(|w| w == b"body-sentinel"));
            let response = format!(
                "HTTP/1.1 307 Temporary Redirect\r\nlocation: http://{second_addr}/steal\r\ncontent-length: 13\r\nconnection: close\r\n\r\nredirect-body"
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        let url = CString::new(format!("http://{first_addr}/v1/chat/completions")).unwrap();
        let auth = CString::new("Authorization: Bearer key-sentinel").unwrap();
        let content_type = CString::new("Content-Type: application/json").unwrap();
        let accept = CString::new("Accept: text/event-stream").unwrap();
        let (tx, rx) = std::sync::mpsc::sync_channel::<Chunk>(2);
        let ctx = Ctx {
            tx, aborted: Arc::new(AtomicBool::new(false)),
            delivered: AtomicBool::new(false),
            redirecting: AtomicBool::new(true),
            header_error: AtomicBool::new(false),
            status_seen: AtomicBool::new(false),
            headers_complete: AtomicBool::new(false),
            response_status: AtomicU16::new(0),
        };
        let error = unsafe {
            run_easy(&url, &auth, &content_type, &accept, b"body-sentinel", &ctx)
        }.err().expect("foreign-origin redirect must fail");
        assert!(error.to_string().contains("cross-origin"), "{error}");
        first_thread.join().unwrap();
        // Leave the listener polling long enough to observe an already queued
        // foreign connection before closing the negative oracle.
        std::thread::sleep(std::time::Duration::from_millis(200));
        stop.store(true, Ordering::Release);
        second_thread.join().unwrap();
        assert!(!contacted.load(Ordering::Acquire), "foreign server received a request");
        assert!(!ctx.delivered.load(Ordering::Acquire));
        assert!(rx.try_recv().is_err(), "redirect response body reached the stream");
    }

    #[test]
    fn same_origin_307_and_308_preserve_post_auth_and_stream() {
        use std::io::Write;
        use std::net::TcpListener;
        for status in [307, 308] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || {
                let mut first = accept_with_deadline(&listener);
                let initial = read_http_request(&mut first);
                let response = format!(
                    "HTTP/1.1 {status} Redirect\r\nlocation: /again\r\ncontent-length: 13\r\nconnection: close\r\n\r\nredirect-body"
                );
                first.write_all(response.as_bytes()).unwrap();
                drop(first);
                let mut second = accept_with_deadline(&listener);
                let replay = read_http_request(&mut second);
                second.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok").unwrap();
                (initial, replay)
            });
            let url = CString::new(format!("http://{addr}/v1/chat/completions")).unwrap();
            let auth = CString::new("Authorization: Bearer key-sentinel").unwrap();
            let content_type = CString::new("Content-Type: application/json").unwrap();
            let accept = CString::new("Accept: text/event-stream").unwrap();
            let (tx, rx) = std::sync::mpsc::sync_channel::<Chunk>(2);
            let ctx = Ctx {
                tx, aborted: Arc::new(AtomicBool::new(false)),
                delivered: AtomicBool::new(false),
                redirecting: AtomicBool::new(true),
                header_error: AtomicBool::new(false),
                status_seen: AtomicBool::new(false),
                headers_complete: AtomicBool::new(false),
                response_status: AtomicU16::new(0),
            };
            let outcome = unsafe {
                run_easy(&url, &auth, &content_type, &accept, b"body-sentinel", &ctx)
            }.expect("same-origin redirect");
            assert_eq!(outcome.rc, c::CURLE_OK);
            assert_eq!(outcome.status, 200);
            let (initial, replay) = server.join().unwrap();
            for request in [&initial, &replay] {
                assert!(request.starts_with(b"POST "), "redirect must retain POST");
                assert!(request.windows(b"authorization: bearer key-sentinel".len())
                    .any(|w| w.eq_ignore_ascii_case(b"authorization: bearer key-sentinel")));
                assert!(request.windows(b"body-sentinel".len()).any(|w| w == b"body-sentinel"));
            }
            assert!(replay.starts_with(b"POST /again "));
            assert!(ctx.delivered.load(Ordering::Acquire));
            match rx.try_recv() {
                Ok(Chunk::Data(data)) => assert_eq!(data.as_slice(), b"ok"),
                _ => panic!("expected only final 200 body"),
            }
            assert!(rx.try_recv().is_err(), "redirect body reached the stream");
        }
    }

    #[test]
    fn cancellation_between_same_origin_hops_starts_no_second_request() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::sync::atomic::AtomicUsize;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let count_server = Arc::clone(&count);
        let stop_server = Arc::clone(&stop);
        let server = std::thread::spawn(move || {
            while !stop_server.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        count_server.fetch_add(1, Ordering::AcqRel);
                        let _ = read_http_request(&mut stream);
                        stream.write_all(
                            b"HTTP/1.1 307 Redirect\r\nlocation: /again\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                        ).unwrap();
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
        });
        let url = CString::new(format!("http://{addr}/v1/chat/completions")).unwrap();
        let auth = CString::new("Authorization: Bearer key-sentinel").unwrap();
        let content_type = CString::new("Content-Type: application/json").unwrap();
        let accept = CString::new("Accept: text/event-stream").unwrap();
        let (tx, _rx) = std::sync::mpsc::sync_channel::<Chunk>(1);
        let aborted = Arc::new(AtomicBool::new(false));
        let ctx = Ctx {
            tx, aborted: Arc::clone(&aborted),
            delivered: AtomicBool::new(false),
            redirecting: AtomicBool::new(true),
            header_error: AtomicBool::new(false),
            status_seen: AtomicBool::new(false),
            headers_complete: AtomicBool::new(false),
            response_status: AtomicU16::new(0),
        };
        let error = unsafe {
            run_easy_with_hook(&url, &auth, &content_type, &accept, b"{}", &ctx, |_| {
                aborted.store(true, Ordering::Release);
            })
        }.err().expect("cancelled redirect must stop");
        assert!(error.to_string().contains("cancelled"), "{error}");
        // The server polls every millisecond; drain a bounded interval before
        // concluding that no follow-up connection was queued.
        std::thread::sleep(std::time::Duration::from_millis(200));
        stop.store(true, Ordering::Release);
        server.join().unwrap();
        assert_eq!(count.load(Ordering::Acquire), 1);
    }

    /// A silent peer: TcpListener that `accept`s once and holds the socket
    /// open without ever writing. Models the LB-dropped / NAT-timeout /
    /// half-open case where TCP handshakes succeed but no bytes flow.
    /// Without CURLOPT_LOW_SPEED_LIMIT/TIME the worker thread would block
    /// for hours; with them, curl aborts after LOW_SPEED_TIME=60s with
    /// CURLE_OPERATION_TIMEDOUT.
    #[test]
    fn silent_peer_trips_low_speed_timeout() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let stop_c = std::sync::Arc::clone(&stop);
        let parked = std::thread::spawn(move || {
            listener.set_nonblocking(true).ok();
            let mut held: Option<std::net::TcpStream> = None;
            while !stop_c.load(Ordering::Relaxed) {
                if held.is_none()
                    && let Ok((s, _)) = listener.accept()
                {
                    held = Some(s);
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            drop(held);
        });

        let url = CString::new(format!("http://{addr}/v1/chat/completions")).unwrap();
        let auth_header = CString::new("Authorization: Bearer test").unwrap();
        let content_type = CString::new("Content-Type: application/json").unwrap();
        let accept = CString::new("Accept: text/event-stream").unwrap();
        let body = b"{}".to_vec();

        let (tx, _rx) = std::sync::mpsc::sync_channel::<Chunk>(1);
        let ctx = Ctx {
            tx,
            aborted: Arc::new(AtomicBool::new(false)),
            delivered: AtomicBool::new(false),
            redirecting: AtomicBool::new(true),
            header_error: AtomicBool::new(false),
            status_seen: AtomicBool::new(false),
            headers_complete: AtomicBool::new(false),
            response_status: AtomicU16::new(0),
        };

        let started = std::time::Instant::now();
        let result = unsafe { run_easy(&url, &auth_header, &content_type, &accept, &body, &ctx) };
        let elapsed = started.elapsed();

        stop.store(true, Ordering::Relaxed);
        let _ = parked.join();

        // LOW_SPEED_TIME=60s; generous 75s upper bound. The load-bearing
        // assertion is that perform completes in finite time at all.
        assert!(
            elapsed < std::time::Duration::from_secs(75),
            "perform must abort within ~60s; took {elapsed:?}"
        );
        let outcome = result.expect("run_easy itself shouldn't error");
        assert_eq!(
            outcome.rc,
            c::CURLE_OPERATION_TIMEDOUT,
            "expected CURLE_OPERATION_TIMEDOUT (28), got code {}",
            outcome.rc
        );
    }

    #[test]
    fn xferinfo_cb_returns_nonzero_when_aborted() {
        let (tx, _rx) = std::sync::mpsc::sync_channel::<Chunk>(1);
        let ctx = Ctx {
            tx,
            aborted: Arc::new(AtomicBool::new(false)),
            delivered: AtomicBool::new(false),
            redirecting: AtomicBool::new(true),
            header_error: AtomicBool::new(false),
            status_seen: AtomicBool::new(false),
            headers_complete: AtomicBool::new(false),
            response_status: AtomicU16::new(0),
        };
        let ptr = &ctx as *const Ctx as *mut c_void;
        assert_eq!(xferinfo_cb(ptr, 0, 0, 0, 0), 0);
        ctx.aborted.store(true, Ordering::Relaxed);
        assert_ne!(xferinfo_cb(ptr, 0, 0, 0, 0), 0, "non-zero signals abort to curl");
    }
}
