//! Conservative HTTP(S) origin comparison for secret-bearing redirects.
//! Unknown authority syntax is an error before sending credentials.

use std::net::Ipv6Addr;

#[derive(Debug, Eq, PartialEq)]
struct Origin {
    scheme: &'static str,
    host: String,
    port: u16,
}

/// Return whether two absolute HTTP(S) URLs share scheme, host and effective
/// port. A malformed or ambiguous authority returns Err, never a guessed
/// equivalence. Redirect destinations are resolved to absolute URLs by
/// libcurl's CURLINFO_REDIRECT_URL before reaching this function.
pub fn same_http_origin(initial: &str, next: &str) -> Result<bool, &'static str> {
    Ok(parse(initial)? == parse(next)?)
}

/// Check an initial request URL before attaching secret headers.
pub fn valid_http_origin(url: &str) -> Result<(), &'static str> {
    parse(url).map(|_| ())
}

fn parse(url: &str) -> Result<Origin, &'static str> {
    let (scheme, rest) = url.split_once("://").ok_or("URL lacks scheme")?;
    let (scheme, default_port) = if scheme.eq_ignore_ascii_case("https") {
        ("https", 443)
    } else if scheme.eq_ignore_ascii_case("http") {
        ("http", 80)
    } else {
        return Err("URL is not HTTP(S)");
    };
    let authority = rest.split(|ch| matches!(ch, '/' | '?' | '#')).next().unwrap_or("");
    if authority.is_empty()
        || authority.bytes().any(|b| b <= 0x20 || b >= 0x7f || b == b'@' || b == b'%' || b == b'\\')
    {
        return Err("invalid or ambiguous URL authority");
    }

    let (host, explicit_port) = if let Some(ipv6) = authority.strip_prefix('[') {
        let (addr, tail) = ipv6.split_once(']').ok_or("unclosed IPv6 host")?;
        let ip = addr.parse::<Ipv6Addr>().map_err(|_| "invalid IPv6 host")?;
        let port = if tail.is_empty() {
            None
        } else {
            Some(tail.strip_prefix(':').ok_or("invalid IPv6 authority")?)
        };
        (ip.to_string(), port)
    } else {
        let (host, port) = match authority.split_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (authority, None),
        };
        if host.is_empty()
            || host.starts_with('.')
            || host.ends_with('.')
            || host.contains("..")
            || !host.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-')
        {
            return Err("invalid DNS host");
        }
        (host.to_ascii_lowercase(), port)
    };
    let port = match explicit_port {
        Some(text) => {
            if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
                return Err("invalid URL port");
            }
            text.parse::<u16>().map_err(|_| "invalid URL port")?
        }
        None => default_port,
    };
    if port == 0 {
        return Err("zero URL port");
    }
    Ok(Origin { scheme, host, port })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_origin_normalizes_case_and_default_port() {
        assert_eq!(same_http_origin("https://Example.com/a", "https://example.com:443/b"), Ok(true));
        assert_eq!(same_http_origin("http://[::1]/a", "http://[0:0:0:0:0:0:0:1]:80/b"), Ok(true));
    }

    #[test]
    fn different_origin_rejects_port_host_and_downgrade() {
        assert_eq!(same_http_origin("http://127.0.0.1:1001/a", "http://127.0.0.1:1002/b"), Ok(false));
        assert_eq!(same_http_origin("https://a.example/a", "https://b.example/b"), Ok(false));
        assert_eq!(same_http_origin("https://a.example/a", "http://a.example/b"), Ok(false));
    }

    #[test]
    fn ambiguous_authority_fails_closed() {
        for url in ["https://u@a.example/", "https://a.example%2f.evil/", "ftp://a.example/",
            "https://a.example:0/", "https://a.example:65536/", "https://a.example\\.evil/"] {
            assert!(valid_http_origin(url).is_err(), "{url}");
        }
    }
}
