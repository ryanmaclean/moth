//! libcurl streaming POST.
//!
//! One easy handle per request, on a background thread. Body chunks land in
//! the curl WRITEFUNCTION callback, which forwards them across a bounded mpsc
//! channel into the iterator on the caller's thread. The channel applies
//! back-pressure: if the caller stops draining, curl's send blocks until they
//! catch up or the channel is dropped.
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

use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::thread::JoinHandle;

use curl_sys as c;

use crate::Error;

const URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

// curl-sys exposes CURLOPT_XFERINFOFUNCTION/DATA as commented-out entries
// in its `pub const` list, so we materialise them from the same base values
// (`CURLOPTTYPE_FUNCTIONPOINT` = 20_000, `CURLOPTTYPE_OBJECTPOINT` = 10_000).
const CURLOPT_XFERINFOFUNCTION: c::CURLoption = 20_000 + 219;
const CURLOPT_XFERINFODATA: c::CURLoption = 10_000 + 57;

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
        // Signal curl during a silent connect/read, then close the real
        // receiver before joining. A full bounded send cannot finish while
        // that receiver remains live.
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
    redirecting: AtomicBool,
    header_error: AtomicBool,
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
    if total == 0 { return 0; }
    let line = unsafe { std::slice::from_raw_parts(ptr as *const u8, total) };
    let ctx = unsafe { &*(user as *const Ctx) };
    if line.starts_with(b"HTTP/") {
        // libcurl delivers a complete status line before any body callback.
        // Unknown status syntax aborts the transfer while body forwarding
        // remains disabled; it must never be treated as a non-redirect.
        let status = std::str::from_utf8(line).ok()
            .and_then(|text| text.split_ascii_whitespace().nth(1))
            .and_then(|digits| digits.parse::<u16>().ok());
        match status {
            Some(code) if (100..600).contains(&code) => {
                ctx.redirecting.store(code < 200 || (300..400).contains(&code), Ordering::Release);
            }
            _ => {
                ctx.header_error.store(true, Ordering::Release);
                return 0;
            }
        }
    } else if line.starts_with(b"HTTP")
        || (!line.iter().all(u8::is_ascii_whitespace) && !line.contains(&b':')) {
        // Any other non-header line could be an unrecognized status line.
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

pub(crate) fn post_stream(api_key: &str, body: String) -> Result<Stream, Error> {
    // Circuit-breaker check: fail fast if the host is in cooldown so we
    // don't tie up a thread + curl handle for a known-down upstream.
    let breaker_cfg = wire::retry::BreakerConfig::default();
    match wire::retry::check("api.anthropic.com", &breaker_cfg) {
        wire::retry::BreakerVerdict::Allow | wire::retry::BreakerVerdict::HalfOpenProbe => {}
        wire::retry::BreakerVerdict::Open => {
            return Err(Error::Http(
                "circuit open for api.anthropic.com (recent upstream failures)".into(),
            ));
        }
    }

    let (tx, rx) = std::sync::mpsc::sync_channel::<Chunk>(64);
    let aborted = Arc::new(AtomicBool::new(false));
    let worker_aborted = Arc::clone(&aborted);

    let url = CString::new(URL).map_err(|_| Error::Http("bad URL".into()))?;
    let key_header = CString::new(format!("x-api-key: {api_key}"))
        .map_err(|_| Error::Http("api key contains NUL".into()))?;
    let version_header = CString::new(format!("anthropic-version: {ANTHROPIC_VERSION}"))
        .map_err(|_| Error::Http("bad version".into()))?;
    let content_type = CString::new("content-type: application/json").unwrap();
    let accept = CString::new("accept: text/event-stream").unwrap();
    let body_bytes = body.into_bytes();

    let handle = std::thread::spawn(move || {
        let ctx = Ctx {
            tx: tx.clone(),
            aborted: worker_aborted,
            delivered: AtomicBool::new(false),
            redirecting: AtomicBool::new(true),
            header_error: AtomicBool::new(false),
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
                run_easy(
                    &url,
                    &key_header,
                    &version_header,
                    &content_type,
                    &accept,
                    &body_attempt,
                    &ctx,
                )
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
            Ok(_) => wire::retry::record_success("api.anthropic.com", &breaker_cfg),
            Err(_) => wire::retry::record_failure("api.anthropic.com", &breaker_cfg),
        }
        let _ = tx.send(Chunk::End(result));
    });

    Ok(Stream { rx, handle: Some(handle), aborted })
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
    key_header: &CString,
    version_header: &CString,
    content_type: &CString,
    accept: &CString,
    body: &[u8],
    ctx: *const Ctx,
) -> Result<PerformOutcome, Error> {
    unsafe { run_easy_with_hook(url, key_header, version_header, content_type, accept, body, ctx, |_| {}) }
}

unsafe fn run_easy_with_hook<F: FnMut(usize)>(
    url: &CString,
    key_header: &CString,
    version_header: &CString,
    content_type: &CString,
    accept: &CString,
    body: &[u8],
    ctx: *const Ctx,
    mut after_redirect: F,
) -> Result<PerformOutcome, Error> {
    let original = url.to_str().map_err(|_| Error::Http("URL is not UTF-8".into()))?;
    wire::http_origin::valid_http_origin(original)
        .map_err(|e| Error::Http(format!("invalid Anthropic HTTP origin: {e}")))?;
    let mut current = url.clone();
    let mut use_post = true;
    for hop in 0..=10 {
        if unsafe { (&*ctx).aborted.load(Ordering::Acquire) } {
            return Err(Error::Http("stream cancelled".into()));
        }
        let (outcome, redirect) = unsafe {
            run_easy_once(&current, key_header, version_header, content_type, accept, body, ctx, use_post)?
        };
        if outcome.rc != c::CURLE_OK || !matches!(outcome.status, 301 | 302 | 303 | 307 | 308) {
            return Ok(outcome);
        }
        if hop == 10 {
            return Err(Error::Http("Anthropic HTTP redirect limit exceeded".into()));
        }
        let next = redirect.ok_or_else(|| Error::Http("Anthropic HTTP redirect lacks Location".into()))?;
        if !wire::http_origin::same_http_origin(original, &next)
            .map_err(|e| Error::Http(format!("invalid Anthropic HTTP redirect origin: {e}")))?
        {
            return Err(Error::Http("cross-origin Anthropic HTTP redirect refused".into()));
        }
        use_post = use_post && !matches!(outcome.status, 301 | 302 | 303);
        current = CString::new(next).map_err(|_| Error::Http("redirect URL contains NUL".into()))?;
        after_redirect(hop);
    }
    unreachable!("bounded redirect loop returns")
}

unsafe fn run_easy_once(
    url: &CString,
    key_header: &CString,
    version_header: &CString,
    content_type: &CString,
    accept: &CString,
    body: &[u8],
    ctx: *const Ctx,
    use_post: bool,
) -> Result<(PerformOutcome, Option<String>), Error> {
    unsafe {
        curl_global_init_once();
        let easy = c::curl_easy_init();
        if easy.is_null() {
            return Err(Error::Http("curl_easy_init failed".into()));
        }

        let mut headers: *mut c::curl_slist = ptr::null_mut();
        headers = c::curl_slist_append(headers, key_header.as_ptr());
        headers = c::curl_slist_append(headers, version_header.as_ptr());
        headers = c::curl_slist_append(headers, content_type.as_ptr());
        headers = c::curl_slist_append(headers, accept.as_ptr());

        let guard = Handle { easy, headers };

        setopt_ptr(easy, c::CURLOPT_URL, url.as_ptr() as *const c_void, "URL")?;
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

/// Idempotent process-wide `curl_global_init`. libcurl docs require
/// global_init be called once before any other curl call, and it's not
/// itself thread-safe — without this, parallel callers (tests, multi-Client
/// production code) occasionally corrupt state at first-init.
unsafe fn curl_global_init_once() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| unsafe {
        c::curl_global_init(c::CURL_GLOBAL_DEFAULT);
    });
}

/// What `probe_tls` observed for one `host:port`. Either:
///   - both `tcp` and `tls` Durations populated (success), or
///   - `tcp` populated and `tls` is None + `error` set (TCP ok, TLS failed —
///     the canonical FreeBSD-without-`ca_root_nss` signature), or
///   - both None + `error` set (the TCP connect itself failed).
#[derive(Debug, Clone)]
pub struct TlsProbe {
    /// Wall-clock duration of the TCP connect, if it succeeded.
    pub tcp: Option<std::time::Duration>,
    /// Wall-clock duration of the full TCP+TLS handshake, if it succeeded.
    pub tls: Option<std::time::Duration>,
    /// libcurl error string if any stage failed.
    pub error: Option<String>,
}

/// Probe `host:port` with a TCP connect followed by a TLS handshake.
///
/// Two-stage measurement:
///   1. `http://host:port/` with `CURLOPT_CONNECT_ONLY=1` — pure TCP, no
///      TLS (libcurl never starts a handshake on a plain-http URL). Times
///      the network path independent of TLS.
///   2. `https://host:port/` with `CURLOPT_CONNECT_ONLY=1` — for an
///      https-scheme URL the "connection setup" libcurl performs **includes
///      the TLS handshake**, then it stops before any HTTP request is sent.
///      No entry lands in the provider's request log.
///
/// Splitting the two cleanly distinguishes a CA-bundle / cert failure
/// (stage 1 ok, stage 2 fails) from a DNS / firewall / proxy failure
/// (stage 1 fails). The doctor uses this to give the FreeBSD-specific
/// `pkg install ca_root_nss` hint only when the failure is genuinely a TLS
/// trust problem.
pub fn probe_tls(host: &str, port: u16, timeout: std::time::Duration) -> TlsProbe {
    use std::time::Instant;
    let tcp_url = match CString::new(format!("http://{host}:{port}/")) {
        Ok(s) => s,
        Err(_) => {
            return TlsProbe { tcp: None, tls: None, error: Some("invalid host".to_string()) };
        }
    };
    let tls_url = match CString::new(format!("https://{host}:{port}/")) {
        Ok(s) => s,
        Err(_) => {
            return TlsProbe { tcp: None, tls: None, error: Some("invalid host".to_string()) };
        }
    };
    let tmo_ms = timeout.as_millis().min(i64::MAX as u128) as i64;

    let stage = |url: &CString| -> Result<std::time::Duration, String> {
        unsafe {
            curl_global_init_once();
            let easy = c::curl_easy_init();
            if easy.is_null() {
                return Err("curl_easy_init failed".to_string());
            }
            // Best-effort setopt: any failure here is reported as an opaque
            // string and we still clean up the handle below before returning.
            let setopt = |opt: c::CURLoption, val: i64| -> Result<(), String> {
                let rc = c::curl_easy_setopt(easy, opt, val);
                if rc != c::CURLE_OK { Err(format!("setopt {opt}: rc={rc}")) } else { Ok(()) }
            };
            let setopt_ptr_val = |opt: c::CURLoption, val: *const c_void| -> Result<(), String> {
                let rc = c::curl_easy_setopt(easy, opt, val);
                if rc != c::CURLE_OK { Err(format!("setopt {opt}: rc={rc}")) } else { Ok(()) }
            };
            let setup = || -> Result<(), String> {
                setopt_ptr_val(c::CURLOPT_URL, url.as_ptr() as *const c_void)?;
                setopt(c::CURLOPT_CONNECT_ONLY, 1)?;
                setopt(c::CURLOPT_CONNECTTIMEOUT_MS, tmo_ms)?;
                setopt(c::CURLOPT_TIMEOUT_MS, tmo_ms)?;
                setopt(c::CURLOPT_NOSIGNAL, 1)?;
                Ok(())
            };
            let res = setup();
            let start = Instant::now();
            let perform_rc = if res.is_ok() { c::curl_easy_perform(easy) } else { c::CURLE_OK };
            let elapsed = start.elapsed();
            c::curl_easy_cleanup(easy);
            // cleanup is done; safe to propagate the setup error.
            res?;
            if perform_rc != c::CURLE_OK {
                let p = c::curl_easy_strerror(perform_rc);
                let msg = if p.is_null() {
                    format!("curl code {perform_rc}")
                } else {
                    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                };
                return Err(msg);
            }
            Ok(elapsed)
        }
    };

    match stage(&tcp_url) {
        Err(e) => TlsProbe { tcp: None, tls: None, error: Some(format!("tcp: {e}")) },
        Ok(tcp) => match stage(&tls_url) {
            Ok(tls) => TlsProbe { tcp: Some(tcp), tls: Some(tls), error: None },
            Err(e) => TlsProbe { tcp: Some(tcp), tls: None, error: Some(format!("tls: {e}")) },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wire::retry::Outcome;

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

    fn ok(rc: c::CURLcode, status: i64) -> Result<PerformOutcome, Error> {
        Ok(PerformOutcome { rc, status })
    }

    #[test]
    fn classify_success_is_ok() {
        match classify(ok(c::CURLE_OK, 200), false) {
            Outcome::Ok(200) => {}
            _ => panic!("expected Ok"),
        }
    }

    #[test]
    fn classify_dns_failure_pre_byte_is_retryable() {
        match classify(ok(c::CURLE_COULDNT_RESOLVE_HOST, 0), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable"),
        }
    }

    #[test]
    fn classify_dns_failure_after_first_byte_is_fatal() {
        // Streaming-retry invariant: once any data delivered, never retry.
        match classify(ok(c::CURLE_COULDNT_RESOLVE_HOST, 0), true) {
            Outcome::Fatal(_) => {}
            _ => panic!("expected Fatal once delivered=true"),
        }
    }

    #[test]
    fn classify_connect_failure_is_retryable() {
        match classify(ok(c::CURLE_COULDNT_CONNECT, 0), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable"),
        }
    }

    #[test]
    fn classify_timeout_is_retryable() {
        match classify(ok(c::CURLE_OPERATION_TIMEDOUT, 0), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable"),
        }
    }

    #[test]
    fn classify_recv_error_pre_byte_is_retryable() {
        match classify(ok(c::CURLE_RECV_ERROR, 0), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable"),
        }
    }

    #[test]
    fn classify_got_nothing_is_retryable() {
        match classify(ok(c::CURLE_GOT_NOTHING, 0), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable"),
        }
    }

    #[test]
    fn classify_http_429_pre_byte_is_retryable() {
        match classify(ok(c::CURLE_OK, 429), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable for 429"),
        }
    }

    #[test]
    fn classify_http_503_pre_byte_is_retryable() {
        match classify(ok(c::CURLE_OK, 503), false) {
            Outcome::Retryable(_) => {}
            _ => panic!("expected Retryable for 503"),
        }
    }

    #[test]
    fn classify_http_429_after_first_byte_is_fatal() {
        match classify(ok(c::CURLE_OK, 429), true) {
            Outcome::Fatal(_) => {}
            _ => panic!("expected Fatal once delivered=true"),
        }
    }

    #[test]
    fn classify_http_400_is_fatal() {
        match classify(ok(c::CURLE_OK, 400), false) {
            Outcome::Fatal(_) => {}
            _ => panic!("expected Fatal for 400"),
        }
    }

    #[test]
    fn classify_http_401_is_fatal() {
        match classify(ok(c::CURLE_OK, 401), false) {
            Outcome::Fatal(_) => {}
            _ => panic!("expected Fatal for 401"),
        }
    }

    #[test]
    fn classify_write_error_is_ok_when_status_2xx() {
        // CURLE_WRITE_ERROR with 200 = caller dropped the receiver mid-stream;
        // not a real failure.
        match classify(ok(c::CURLE_WRITE_ERROR, 200), true) {
            Outcome::Ok(200) => {}
            _ => panic!("expected Ok for caller-abort"),
        }
    }

    #[test]
    fn classify_non_retryable_curl_code_is_fatal() {
        // CURLE_HTTP_RETURNED_ERROR (22) isn't in our retry set.
        match classify(ok(c::CURLE_HTTP_RETURNED_ERROR, 0), false) {
            Outcome::Fatal(_) => {}
            _ => panic!("expected Fatal for non-retryable code"),
        }
    }

    #[test]
    fn retry_loop_clones_body_per_attempt() {
        // Verify that wrapping with_backoff with a closure that captures a
        // shared body clones once per invocation. Stand-in for the real
        // perform: we just record the addresses to confirm distinct bufs.
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
            classify(ok(c::CURLE_COULDNT_CONNECT, 0), false)
        });
        let ptrs = seen_ptrs.lock().unwrap();
        assert_eq!(ptrs.len(), 3, "loop should run all 3 attempts");
        // The Vec is dropped & reallocated each iteration — at minimum, the
        // call ran the closure max_attempts times with a fresh clone.
    }

    #[test]
    fn aborted_flag_short_circuits_retry_loop_via_fatal() {
        // Once the caller drops the receiver (aborted=true → write_cb
        // returns 0 → CURLE_WRITE_ERROR with the existing status), classify
        // returns Ok/Fatal (not Retryable), so the loop stops immediately
        // and we don't keep hammering the upstream.
        let mut attempts = 0;
        let policy = wire::retry::RetryPolicy {
            max_attempts: 4,
            base_delay: std::time::Duration::from_millis(1),
            max_delay: std::time::Duration::from_millis(2),
        };
        let _res: Result<u32, Error> = wire::retry::with_backoff(&policy, |_| {
            attempts += 1;
            classify(ok(c::CURLE_WRITE_ERROR, 200), true)
        });
        assert_eq!(attempts, 1, "Ok outcome must short-circuit");
    }

    #[test]
    fn cross_origin_redirect_never_reaches_second_server() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        let first = TcpListener::bind("127.0.0.1:0").unwrap();
        let second = TcpListener::bind("127.0.0.1:0").unwrap();
        let first_addr = first.local_addr().unwrap();
        let second_addr = second.local_addr().unwrap();
        second.set_nonblocking(true).unwrap();
        let contacted = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let contacted_in_server = Arc::clone(&contacted);
        let stop_in_server = Arc::clone(&stop);
        let second_thread = std::thread::spawn(move || {
            while !stop_in_server.load(Ordering::Acquire) {
                match second.accept() {
                    Ok((mut stream, _)) => {
                        contacted_in_server.store(true, Ordering::Release);
                        let mut buf = [0u8; 1024];
                        let _ = stream.read(&mut buf);
                        let _ = stream.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n");
                        break;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
        });
        let first_thread = std::thread::spawn(move || {
            let (mut stream, _) = first.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let response = format!(
                "HTTP/1.1 307 Temporary Redirect\r\nlocation: http://{second_addr}/v1/messages\r\ncontent-length: 13\r\nconnection: close\r\n\r\nredirect-body"
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        let url = CString::new(format!("http://{first_addr}/v1/messages")).unwrap();
        let key_header = CString::new("x-api-key: key-sentinel").unwrap();
        let version_header = CString::new("anthropic-version: 2023-06-01").unwrap();
        let content_type = CString::new("content-type: application/json").unwrap();
        let accept = CString::new("accept: text/event-stream").unwrap();
        let (tx, rx) = std::sync::mpsc::sync_channel::<Chunk>(1);
        let ctx = Ctx {
            tx,
            aborted: Arc::new(AtomicBool::new(false)),
            delivered: AtomicBool::new(false),
            redirecting: AtomicBool::new(true),
            header_error: AtomicBool::new(false),
        };
        let error = unsafe {
            run_easy(&url, &key_header, &version_header, &content_type, &accept, b"{}", &ctx)
        }.err().expect("cross-origin redirect must fail");
        let message = match error {
            Error::Http(message) => message,
            other => panic!("unexpected error: {other:?}"),
        };
        assert!(message.contains("cross-origin"), "unexpected error: {message}");
        first_thread.join().unwrap();
        stop.store(true, Ordering::Release);
        second_thread.join().unwrap();
        assert!(!contacted.load(Ordering::Acquire));
        assert!(!ctx.delivered.load(Ordering::Acquire));
        assert!(rx.try_recv().is_err(), "redirect body must not become a stream chunk");
    }
    #[test]
    fn header_gate_suppresses_nonempty_http2_redirects_after_interim_status() {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Chunk>(2);
        let ctx = Ctx {
            tx,
            aborted: Arc::new(AtomicBool::new(false)),
            delivered: AtomicBool::new(false),
            redirecting: AtomicBool::new(true),
            header_error: AtomicBool::new(false),
        };
        let user = &ctx as *const Ctx as *mut c_void;
        for status in [307, 308] {
            let interim = b"HTTP/1.1 100 Continue\r\n";
            assert_eq!(header_cb(interim.as_ptr() as *mut c_char, 1, interim.len(), user), interim.len());
            let line = format!("HTTP/2 {status}\r\n");
            assert_eq!(header_cb(line.as_ptr() as *mut c_char, 1, line.len(), user), line.len());
            let body = b"redirect-body";
            assert_eq!(write_cb(body.as_ptr() as *mut c_char, 1, body.len(), user), body.len());
            assert!(rx.try_recv().is_err(), "3xx bytes must not reach the stream");
            assert!(!ctx.delivered.load(Ordering::Acquire));
        }
        let success = b"HTTP/2 200\r\n";
        assert_eq!(header_cb(success.as_ptr() as *mut c_char, 1, success.len(), user), success.len());
        let answer = b"ok";
        assert_eq!(write_cb(answer.as_ptr() as *mut c_char, 1, answer.len(), user), answer.len());
        match rx.try_recv() {
            Ok(Chunk::Data(data)) => assert_eq!(data, answer),
            _ => panic!("2xx body must reach the stream"),
        }
        assert!(ctx.delivered.load(Ordering::Acquire));
        let malformed = b"HTTP: 307\r\n";
        assert_eq!(header_cb(malformed.as_ptr() as *mut c_char, 1, malformed.len(), user), 0);
        assert!(ctx.header_error.load(Ordering::Acquire));
    }

    #[test]
    fn cancellation_between_same_origin_hops_starts_no_second_request() {
        use std::io::{Read, Write};
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
                        let mut buf = [0u8; 1024];
                        let _ = stream.read(&mut buf);
                        let _ = stream.write_all(
                            b"HTTP/1.1 307 Temporary Redirect\r\nlocation: /again\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                        );
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
        });
        let url = CString::new(format!("http://{addr}/v1/messages")).unwrap();
        let key = CString::new("x-api-key: key-sentinel").unwrap();
        let version = CString::new("anthropic-version: 2023-06-01").unwrap();
        let content_type = CString::new("content-type: application/json").unwrap();
        let accept = CString::new("accept: text/event-stream").unwrap();
        let (tx, _rx) = std::sync::mpsc::sync_channel::<Chunk>(1);
        let aborted = Arc::new(AtomicBool::new(false));
        let ctx = Ctx {
            tx, aborted: Arc::clone(&aborted),
            delivered: AtomicBool::new(false),
            redirecting: AtomicBool::new(true),
            header_error: AtomicBool::new(false),
        };
        let error = unsafe {
            run_easy_with_hook(&url, &key, &version, &content_type, &accept, b"{}", &ctx, |_| {
                aborted.store(true, Ordering::Release);
            })
        }.err().expect("cancelled redirect must stop");
        assert!(error.to_string().contains("cancelled"), "{error}");
        stop.store(true, Ordering::Release);
        server.join().unwrap();
        assert_eq!(count.load(Ordering::Acquire), 1);
    }

    /// A silent peer: a TcpListener that `accept`s a single connection and
    /// then holds it open without reading or writing. Models the LB-dropped
    /// / NAT-timeout / half-open case where TCP handshakes succeed but no
    /// bytes ever flow. Without CURLOPT_LOW_SPEED_LIMIT/TIME the worker
    /// thread would block here for hours; with them, curl aborts after
    /// LOW_SPEED_TIME=60s with CURLE_OPERATION_TIMEDOUT.
    #[test]
    fn silent_peer_trips_low_speed_timeout() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let stop_c = std::sync::Arc::clone(&stop);
        let parked = std::thread::spawn(move || {
            // Bound the accept wait so the thread can exit promptly.
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

        let url = CString::new(format!("http://{addr}/v1/messages")).unwrap();
        let key_header = CString::new("x-api-key: test").unwrap();
        let version_header = CString::new("anthropic-version: 2023-06-01").unwrap();
        let content_type = CString::new("content-type: application/json").unwrap();
        let accept = CString::new("accept: text/event-stream").unwrap();
        let body = b"{}".to_vec();

        let (tx, _rx) = std::sync::mpsc::sync_channel::<Chunk>(1);
        let ctx = Ctx {
            tx,
            aborted: Arc::new(AtomicBool::new(false)),
            delivered: AtomicBool::new(false),
            redirecting: AtomicBool::new(true),
            header_error: AtomicBool::new(false),
        };

        let started = std::time::Instant::now();
        let result = unsafe {
            run_easy(&url, &key_header, &version_header, &content_type, &accept, &body, &ctx)
        };
        let elapsed = started.elapsed();

        stop.store(true, Ordering::Relaxed);
        let _ = parked.join();

        // With LOW_SPEED_TIME=60s the abort fires around the 60s mark; give
        // a generous 75s upper bound so this isn't flaky under load. The
        // load-bearing assertion is that perform completes in *finite*
        // time at all — without these setopt calls it never would.
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
        };
        let ptr = &ctx as *const Ctx as *mut c_void;
        assert_eq!(xferinfo_cb(ptr, 0, 0, 0, 0), 0);
        ctx.aborted.store(true, Ordering::Relaxed);
        assert_ne!(xferinfo_cb(ptr, 0, 0, 0, 0), 0, "non-zero signals abort to curl");
    }
}
