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

use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::sync::mpsc::{Receiver, SyncSender};
use std::thread::JoinHandle;

use curl_sys as c;

use crate::Error;

const PATH: &str = "/v1/chat/completions";

pub(crate) enum Chunk {
    Data(Vec<u8>),
    End(Result<u32, Error>),
}

pub(crate) struct Stream {
    pub(crate) rx: Receiver<Chunk>,
    pub(crate) handle: Option<JoinHandle<()>>,
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

struct Ctx {
    tx: SyncSender<Chunk>,
    aborted: bool,
}

extern "C" fn write_cb(
    ptr: *mut c_char,
    size: usize,
    nmemb: usize,
    user: *mut c_void,
) -> usize {
    let total = size.saturating_mul(nmemb);
    if total == 0 {
        return 0;
    }
    let ctx = unsafe { &mut *(user as *mut Ctx) };
    if ctx.aborted {
        return 0;
    }
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, total) };
    match ctx.tx.send(Chunk::Data(slice.to_vec())) {
        Ok(()) => total,
        Err(_) => {
            ctx.aborted = true;
            0
        }
    }
}

pub(crate) fn post_stream(
    api_key: &str,
    base_url: &str,
    body: String,
) -> Result<Stream, Error> {
    let host = host_of(base_url);
    let breaker_cfg = wire::retry::BreakerConfig::default();
    match wire::retry::check(&host, &breaker_cfg) {
        wire::retry::BreakerVerdict::Allow | wire::retry::BreakerVerdict::HalfOpenProbe => {}
        wire::retry::BreakerVerdict::Open => {
            return Err(Error::Http(format!(
                "circuit open for {host} (recent upstream failures)"
            )));
        }
    }

    let (tx, rx) = std::sync::mpsc::sync_channel::<Chunk>(64);

    let full_url = join_url(base_url, PATH);
    let url = CString::new(full_url).map_err(|_| Error::Http("bad URL".into()))?;
    let auth_header = CString::new(format!("Authorization: Bearer {api_key}"))
        .map_err(|_| Error::Http("api key contains NUL".into()))?;
    let content_type = CString::new("Content-Type: application/json").unwrap();
    let accept = CString::new("Accept: text/event-stream").unwrap();
    let body_c = body.into_bytes();

    let handle = std::thread::spawn(move || {
        let mut ctx = Ctx { tx: tx.clone(), aborted: false };
        let result = unsafe {
            run_easy(&url, &auth_header, &content_type, &accept, &body_c, &mut ctx)
        };
        match &result {
            Ok(_) => wire::retry::record_success(&host, &breaker_cfg),
            Err(_) => wire::retry::record_failure(&host, &breaker_cfg),
        }
        let _ = tx.send(Chunk::End(result));
    });

    Ok(Stream { rx, handle: Some(handle) })
}

/// Extract the host portion of `base_url` (e.g. "https://api.openai.com/v1"
/// → "api.openai.com"). Defaults to the whole input if parsing fails so
/// the breaker still keys consistently per caller-supplied URL.
fn host_of(base_url: &str) -> String {
    let after_scheme = base_url
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(base_url);
    after_scheme
        .split(['/', ':', '?'])
        .next()
        .unwrap_or(base_url)
        .to_string()
}

fn join_url(base: &str, path: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    format!("{trimmed}{path}")
}

unsafe fn run_easy(
    url: &CString,
    auth_header: &CString,
    content_type: &CString,
    accept: &CString,
    body: &[u8],
    ctx: *mut Ctx,
) -> Result<u32, Error> {
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
        setopt_long(easy, c::CURLOPT_POST, 1, "POST")?;
        setopt_ptr(
            easy,
            c::CURLOPT_POSTFIELDS,
            body.as_ptr() as *const c_void,
            "POSTFIELDS",
        )?;
        setopt_long(
            easy,
            c::CURLOPT_POSTFIELDSIZE,
            body.len() as i64,
            "POSTFIELDSIZE",
        )?;
        setopt_ptr(
            easy,
            c::CURLOPT_HTTPHEADER,
            headers as *const c_void,
            "HTTPHEADER",
        )?;
        setopt_ptr(
            easy,
            c::CURLOPT_WRITEFUNCTION,
            write_cb as *const c_void,
            "WRITEFUNCTION",
        )?;
        setopt_ptr(
            easy,
            c::CURLOPT_WRITEDATA,
            ctx as *const c_void,
            "WRITEDATA",
        )?;
        setopt_long(easy, c::CURLOPT_NOPROGRESS, 1, "NOPROGRESS")?;
        setopt_long(easy, c::CURLOPT_FOLLOWLOCATION, 1, "FOLLOWLOCATION")?;

        let perform_rc = c::curl_easy_perform(easy);
        let mut status: i64 = 0;
        c::curl_easy_getinfo(easy, c::CURLINFO_RESPONSE_CODE, &mut status);

        drop(guard);

        if perform_rc != c::CURLE_OK && perform_rc != c::CURLE_WRITE_ERROR {
            return Err(curl_err(perform_rc, "perform"));
        }
        if !(200..300).contains(&status) {
            return Err(Error::Http(format!("HTTP {status}")));
        }
        Ok(status as u32)
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
    use super::join_url;

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
}
