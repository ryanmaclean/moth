//! One-shot libcurl request, synchronous.
//!
//! Same shape as `mcp::transport::http::post` and `anthropic::http::run_easy`:
//! one easy handle per request, write callback into a `Vec<u8>`, no async.
//! Gitea uses `Authorization: token <PAT>` (not `Bearer`) — that's set by the
//! caller in the `headers` slice.

use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;

use curl_sys as c;

use crate::Error;

pub(crate) struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}

/// Perform a request. `method` is "GET" or "POST". `body` is the request body
/// (empty for GET). `headers` is a list of pre-formatted "name: value" header
/// lines — the caller is responsible for the authorization header.
pub(crate) fn request(
    url: &str,
    method: &str,
    headers: &[String],
    body: &[u8],
) -> Result<Response, Error> {
    let url_c = CString::new(url).map_err(|_| Error::Http("URL contains NUL".into()))?;
    let method_c = CString::new(method).map_err(|_| Error::Http("bad method".into()))?;

    let mut header_strings: Vec<CString> = Vec::with_capacity(headers.len());
    for h in headers {
        header_strings.push(
            CString::new(h.as_str()).map_err(|_| Error::Http("header contains NUL".into()))?,
        );
    }

    let mut body_buf: Vec<u8> = Vec::new();

    let status: u16;

    unsafe {
        let easy = c::curl_easy_init();
        if easy.is_null() {
            return Err(Error::Http("curl_easy_init failed".into()));
        }

        let mut hdr_list: *mut c::curl_slist = ptr::null_mut();
        for h in &header_strings {
            hdr_list = c::curl_slist_append(hdr_list, h.as_ptr());
        }
        let guard = Handle { easy, headers: hdr_list };

        setopt_ptr(easy, c::CURLOPT_URL, url_c.as_ptr() as *const c_void, "URL")?;
        setopt_ptr(
            easy,
            c::CURLOPT_CUSTOMREQUEST,
            method_c.as_ptr() as *const c_void,
            "CUSTOMREQUEST",
        )?;
        if !body.is_empty() {
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
        }
        setopt_ptr(
            easy,
            c::CURLOPT_HTTPHEADER,
            hdr_list as *const c_void,
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
            (&mut body_buf as *mut Vec<u8>) as *const c_void,
            "WRITEDATA",
        )?;
        setopt_long(easy, c::CURLOPT_NOPROGRESS, 1, "NOPROGRESS")?;
        setopt_long(easy, c::CURLOPT_FOLLOWLOCATION, 1, "FOLLOWLOCATION")?;

        let rc = c::curl_easy_perform(easy);
        let mut code: i64 = 0;
        c::curl_easy_getinfo(easy, c::CURLINFO_RESPONSE_CODE, &mut code);

        drop(guard);

        if rc != c::CURLE_OK {
            return Err(curl_err(rc, "perform"));
        }
        status = code.clamp(0, u16::MAX as i64) as u16;
    }

    Ok(Response { status, body: body_buf })
}

extern "C" fn write_cb(
    ptr: *mut std::os::raw::c_char,
    size: usize,
    nmemb: usize,
    user: *mut c_void,
) -> usize {
    let total = size.saturating_mul(nmemb);
    if total == 0 {
        return 0;
    }
    let buf = unsafe { &mut *(user as *mut Vec<u8>) };
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, total) };
    buf.extend_from_slice(slice);
    total
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
