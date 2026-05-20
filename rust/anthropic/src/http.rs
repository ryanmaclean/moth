//! libcurl streaming POST.
//!
//! One easy handle per request, on a background thread. Body chunks land in
//! the curl WRITEFUNCTION callback, which forwards them across a bounded
//! mpsc channel into the iterator on the caller's thread. The channel applies
//! back-pressure: if the caller stops draining, curl's send blocks until they
//! catch up or the channel is dropped.

use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::sync::mpsc::{Receiver, SyncSender};
use std::thread::JoinHandle;

use curl_sys as c;

use crate::Error;

const URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

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
        // Dropping rx closes the channel; the next write callback returns
        // CURLE_WRITE_ERROR and curl_easy_perform exits.
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

pub(crate) fn post_stream(api_key: &str, body: String) -> Result<Stream, Error> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<Chunk>(64);

    let url = CString::new(URL).map_err(|_| Error::Http("bad URL".into()))?;
    let key_header = CString::new(format!("x-api-key: {api_key}"))
        .map_err(|_| Error::Http("api key contains NUL".into()))?;
    let version_header = CString::new(format!("anthropic-version: {ANTHROPIC_VERSION}"))
        .map_err(|_| Error::Http("bad version".into()))?;
    let content_type = CString::new("content-type: application/json").unwrap();
    let accept = CString::new("accept: text/event-stream").unwrap();
    let body_c = body.into_bytes();

    let handle = std::thread::spawn(move || {
        let mut ctx = Ctx { tx: tx.clone(), aborted: false };
        let result = unsafe { run_easy(&url, &key_header, &version_header, &content_type, &accept, &body_c, &mut ctx) };
        let _ = tx.send(Chunk::End(result));
    });

    Ok(Stream { rx, handle: Some(handle) })
}

#[allow(clippy::too_many_arguments)]
unsafe fn run_easy(
    url: &CString,
    key_header: &CString,
    version_header: &CString,
    content_type: &CString,
    accept: &CString,
    body: &[u8],
    ctx: *mut Ctx,
) -> Result<u32, Error> {
    let easy = unsafe { c::curl_easy_init() };
    if easy.is_null() {
        return Err(Error::Http("curl_easy_init failed".into()));
    }

    let mut headers: *mut c::curl_slist = ptr::null_mut();
    unsafe {
        headers = c::curl_slist_append(headers, key_header.as_ptr());
        headers = c::curl_slist_append(headers, version_header.as_ptr());
        headers = c::curl_slist_append(headers, content_type.as_ptr());
        headers = c::curl_slist_append(headers, accept.as_ptr());
    }

    let result = unsafe {
        let mut rc;
        rc = c::curl_easy_setopt(easy, c::CURLOPT_URL, url.as_ptr());
        if rc != c::CURLE_OK {
            cleanup(easy, headers);
            return Err(curl_err(rc, "setopt URL"));
        }
        rc = c::curl_easy_setopt(easy, c::CURLOPT_POST, 1_i64);
        if rc != c::CURLE_OK {
            cleanup(easy, headers);
            return Err(curl_err(rc, "setopt POST"));
        }
        rc = c::curl_easy_setopt(easy, c::CURLOPT_POSTFIELDS, body.as_ptr());
        if rc != c::CURLE_OK {
            cleanup(easy, headers);
            return Err(curl_err(rc, "setopt POSTFIELDS"));
        }
        rc = c::curl_easy_setopt(easy, c::CURLOPT_POSTFIELDSIZE, body.len() as i64);
        if rc != c::CURLE_OK {
            cleanup(easy, headers);
            return Err(curl_err(rc, "setopt POSTFIELDSIZE"));
        }
        rc = c::curl_easy_setopt(easy, c::CURLOPT_HTTPHEADER, headers);
        if rc != c::CURLE_OK {
            cleanup(easy, headers);
            return Err(curl_err(rc, "setopt HTTPHEADER"));
        }
        rc = c::curl_easy_setopt(easy, c::CURLOPT_WRITEFUNCTION, write_cb as *const c_void);
        if rc != c::CURLE_OK {
            cleanup(easy, headers);
            return Err(curl_err(rc, "setopt WRITEFUNCTION"));
        }
        rc = c::curl_easy_setopt(easy, c::CURLOPT_WRITEDATA, ctx as *mut c_void);
        if rc != c::CURLE_OK {
            cleanup(easy, headers);
            return Err(curl_err(rc, "setopt WRITEDATA"));
        }
        rc = c::curl_easy_setopt(easy, c::CURLOPT_NOPROGRESS, 1_i64);
        if rc != c::CURLE_OK {
            cleanup(easy, headers);
            return Err(curl_err(rc, "setopt NOPROGRESS"));
        }
        rc = c::curl_easy_setopt(easy, c::CURLOPT_FOLLOWLOCATION, 1_i64);
        if rc != c::CURLE_OK {
            cleanup(easy, headers);
            return Err(curl_err(rc, "setopt FOLLOWLOCATION"));
        }

        let perform_rc = c::curl_easy_perform(easy);
        let mut status: i64 = 0;
        let _ = c::curl_easy_getinfo(easy, c::CURLINFO_RESPONSE_CODE, &mut status);
        cleanup(easy, headers);

        // CURLE_WRITE_ERROR is what we return from write_cb when the receiver
        // dropped — that's a clean caller-initiated abort, not a real failure.
        if perform_rc != c::CURLE_OK && perform_rc != c::CURLE_WRITE_ERROR {
            return Err(curl_err(perform_rc, "perform"));
        }
        if status < 200 || status >= 300 {
            return Err(Error::Http(format!("HTTP {status}")));
        }
        Ok(status as u32)
    };

    result
}

unsafe fn cleanup(easy: *mut c::CURL, headers: *mut c::curl_slist) {
    unsafe {
        if !headers.is_null() {
            c::curl_slist_free_all(headers);
        }
        c::curl_easy_cleanup(easy);
    }
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
