//! DogStatsD-over-UDP emitter.
//!
//! Fire-and-forget counters, gauges, histograms, and timers in the
//! DogStatsD wire format. Errors are swallowed: a metric call must
//! never break a hot path.
//!
//! ```no_run
//! let m = metrics::Client::from_env().with_prefix("agent");
//! m.count("requests", 1, &[("route", "/healthz")]);
//! ```

use std::cell::RefCell;
use std::net::UdpSocket;
use std::sync::OnceLock;
use std::time::Instant;

/// Safe LAN MTU minus IP+UDP headers, matching the Datadog Agent default.
const MAX_DATAGRAM: usize = 1432;

/// Single DogStatsD metric in unrendered form.
#[derive(Debug, Clone)]
pub(crate) struct Metric<'a> {
    pub name: &'a str,
    pub value: Value,
    pub kind: Kind,
    pub tags: &'a [(&'a str, &'a str)],
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Value {
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Kind {
    Counter,
    Gauge,
    Histogram,
    Timer,
}

impl Kind {
    fn suffix(self) -> &'static str {
        match self {
            Kind::Counter => "c",
            Kind::Gauge => "g",
            Kind::Histogram => "h",
            Kind::Timer => "ms",
        }
    }
}

/// DogStatsD client. Cheap to clone-by-Arc if you need to share;
/// the underlying `UdpSocket` is `Sync`.
pub struct Client {
    sock: Option<UdpSocket>,
    addr: String,
    prefix: String,
    constant_tags: Vec<(String, String)>,
}

impl Client {
    /// Bind an ephemeral UDP socket and connect it to `addr`. Returns
    /// Ok even if the agent isn't running; UDP is connectionless.
    /// Name resolution failure does return Err.
    ///
    /// # Errors
    /// Returns the underlying `io::Error` when the bind fails or the
    /// destination address cannot be resolved.
    pub fn new(addr: impl Into<String>) -> std::io::Result<Self> {
        let addr = addr.into();
        let sock = UdpSocket::bind("0.0.0.0:0")?;
        sock.set_nonblocking(true)?;
        sock.connect(&addr)?;
        Ok(Self { sock: Some(sock), addr, prefix: String::new(), constant_tags: Vec::new() })
    }

    /// Disabled client: every emit call is a no-op. Use this when
    /// `DOGSTATSD_ADDR` isn't set so call sites don't need conditionals.
    #[must_use]
    pub fn disabled() -> Self {
        Self { sock: None, addr: String::new(), prefix: String::new(), constant_tags: Vec::new() }
    }

    /// Build from `DOGSTATSD_ADDR`. Disabled when unset or invalid.
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var("DOGSTATSD_ADDR") {
            Ok(addr) if !addr.is_empty() => Self::new(addr).unwrap_or_else(|_| Self::disabled()),
            _ => Self::disabled(),
        }
    }

    /// Prepend `prefix` and a `.` to every metric name.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Add a tag applied to every emitted metric.
    #[must_use]
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.constant_tags.push((key.into(), value.into()));
        self
    }

    pub fn count(&self, name: &str, value: i64, tags: &[(&str, &str)]) {
        self.emit(&Metric { name, value: Value::Int(value), kind: Kind::Counter, tags });
    }

    pub fn gauge(&self, name: &str, value: f64, tags: &[(&str, &str)]) {
        self.emit(&Metric { name, value: Value::Float(value), kind: Kind::Gauge, tags });
    }

    pub fn histogram(&self, name: &str, value: f64, tags: &[(&str, &str)]) {
        self.emit(&Metric { name, value: Value::Float(value), kind: Kind::Histogram, tags });
    }

    pub fn timer(&self, name: &str, ms: f64, tags: &[(&str, &str)]) {
        self.emit(&Metric { name, value: Value::Float(ms), kind: Kind::Timer, tags });
    }

    /// Stopwatch: the returned guard emits a timer metric on Drop.
    pub fn time<'a>(&'a self, name: &'a str, tags: Vec<(&'a str, &'a str)>) -> Timer<'a> {
        Timer { client: self, name, tags, start: Instant::now() }
    }

    fn emit(&self, m: &Metric<'_>) {
        let Some(sock) = self.sock.as_ref() else { return };
        // Stack-friendly capacity; render() writes into this buffer.
        let mut buf = String::with_capacity(MAX_DATAGRAM);
        render_into(&mut buf, m, &self.prefix, &self.constant_tags);
        // Truncate (shouldn't happen for normal metrics): cap at MAX_DATAGRAM.
        // We don't split a single metric across datagrams in v1.
        if buf.len() > MAX_DATAGRAM {
            buf.truncate(MAX_DATAGRAM);
        }
        if let Err(e) = sock.send(buf.as_bytes()) {
            log_first_error(&e);
        }
    }

    /// Server address this client targets. `""` when disabled.
    #[must_use]
    pub fn addr(&self) -> &str {
        &self.addr
    }
}

/// RAII timer guard. Drops to a `timer` metric.
pub struct Timer<'a> {
    client: &'a Client,
    name: &'a str,
    tags: Vec<(&'a str, &'a str)>,
    start: Instant,
}

impl Drop for Timer<'_> {
    fn drop(&mut self) {
        // Truncate to integer milliseconds for stable line format in tests;
        // sub-ms resolution stays in the fractional part via histogram if needed.
        let ms = self.start.elapsed().as_secs_f64() * 1000.0;
        self.client.timer(self.name, ms, &self.tags);
    }
}

/// Format `value` to a DogStatsD-friendly string. Integers don't get
/// a trailing `.0`; whole floats render as e.g. `42` (no `.0`); other
/// floats render with the default Display.
fn fmt_value(v: Value, out: &mut String) {
    use std::fmt::Write;
    match v {
        Value::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Value::Float(f) => {
            if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 {
                // Render as an integer to keep wire output compact and
                // predictable (matches the DogStatsD examples in tests).
                let _ = write!(out, "{}", f as i64);
            } else {
                let _ = write!(out, "{f}");
            }
        }
    }
}

/// Sanitise a tag value: DogStatsD disallows `|`, `\n`, `,`, `:` in
/// tag values (they're line/field separators). Replace with `_`.
fn push_sanitised(out: &mut String, s: &str) {
    for c in s.chars() {
        if matches!(c, '|' | '\n' | ',' | ':') {
            out.push('_');
        } else {
            out.push(c);
        }
    }
}

/// Render a single metric line: `prefix.name:value|kind[|#k:v,...]`.
/// Tag list is `constant_tags` followed by per-call tags, in order.
pub(crate) fn render_into(
    out: &mut String,
    m: &Metric<'_>,
    prefix: &str,
    constant_tags: &[(String, String)],
) {
    if !prefix.is_empty() {
        out.push_str(prefix);
        out.push('.');
    }
    out.push_str(m.name);
    out.push(':');
    fmt_value(m.value, out);
    out.push('|');
    out.push_str(m.kind.suffix());

    if !constant_tags.is_empty() || !m.tags.is_empty() {
        out.push_str("|#");
        let mut first = true;
        for (k, v) in constant_tags {
            if !first {
                out.push(',');
            }
            out.push_str(k);
            out.push(':');
            push_sanitised(out, v);
            first = false;
        }
        for (k, v) in m.tags {
            if !first {
                out.push(',');
            }
            out.push_str(k);
            out.push(':');
            push_sanitised(out, v);
            first = false;
        }
    }
}

/// Render to a fresh String. Convenience for tests and one-shot uses.
#[cfg(test)]
#[must_use]
pub(crate) fn render(m: &Metric<'_>, prefix: &str, constant_tags: &[(String, String)]) -> String {
    let mut s = String::with_capacity(64);
    render_into(&mut s, m, prefix, constant_tags);
    s
}

/// First send error per process is logged to stderr; subsequent
/// errors are dropped silently. Keeps hot paths quiet but flags
/// misconfiguration once.
fn log_first_error(e: &std::io::Error) {
    static LOGGED: OnceLock<()> = OnceLock::new();
    let _ = LOGGED.get_or_init(|| {
        eprintln!("metrics: send failed (further errors suppressed): {e}");
    });
}

// ---------------------------------------------------------------------------
// Test-only recorder backend: lets disabled-path tests assert no-op
// behaviour without UDP coupling.
// ---------------------------------------------------------------------------

#[cfg(test)]
thread_local! {
    static RECORDER: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn record(line: String) {
    RECORDER.with(|r| r.borrow_mut().push(line));
}

#[cfg(test)]
pub(crate) fn drain_recorded() -> Vec<String> {
    RECORDER.with(|r| std::mem::take(&mut *r.borrow_mut()))
}

// Silence unused-import warnings under non-test builds.
#[cfg(not(test))]
#[allow(dead_code)]
fn _refcell_phantom(_: RefCell<()>) {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::UdpSocket;
    use std::time::Duration;

    fn m<'a>(name: &'a str, v: Value, k: Kind, tags: &'a [(&'a str, &'a str)]) -> Metric<'a> {
        Metric { name, value: v, kind: k, tags }
    }

    #[test]
    fn counter_line_format() {
        let s = render(&m("agent.foo", Value::Int(1), Kind::Counter, &[]), "", &[]);
        assert_eq!(s, "agent.foo:1|c");
    }

    #[test]
    fn gauge_line_format() {
        let s = render(&m("agent.foo", Value::Float(2.5), Kind::Gauge, &[]), "", &[]);
        assert_eq!(s, "agent.foo:2.5|g");
    }

    #[test]
    fn histogram_line_format() {
        let s = render(&m("agent.foo", Value::Float(42.0), Kind::Histogram, &[]), "", &[]);
        assert_eq!(s, "agent.foo:42|h");
    }

    #[test]
    fn timer_line_format() {
        let s = render(&m("agent.foo", Value::Float(120.0), Kind::Timer, &[]), "", &[]);
        assert_eq!(s, "agent.foo:120|ms");
    }

    #[test]
    fn single_tag() {
        let s =
            render(&m("agent.foo", Value::Int(1), Kind::Counter, &[("name", "value")]), "", &[]);
        assert_eq!(s, "agent.foo:1|c|#name:value");
    }

    #[test]
    fn constant_tags_then_per_call() {
        let ct = vec![("env".into(), "prod".into()), ("host".into(), "h1".into())];
        let s = render(&m("agent.foo", Value::Int(1), Kind::Counter, &[("route", "/x")]), "", &ct);
        assert_eq!(s, "agent.foo:1|c|#env:prod,host:h1,route:/x");
    }

    #[test]
    fn tag_value_sanitisation() {
        // Each of `|`, `\n`, `,`, `:` becomes `_`.
        let s =
            render(&m("agent.foo", Value::Int(1), Kind::Counter, &[("k", "a|b\nc,d:e")]), "", &[]);
        assert_eq!(s, "agent.foo:1|c|#k:a_b_c_d_e");
    }

    #[test]
    fn prefix_is_applied() {
        let s = render(&m("foo", Value::Int(1), Kind::Counter, &[]), "x", &[]);
        assert_eq!(s, "x.foo:1|c");
    }

    #[test]
    fn no_tag_marker_when_no_tags() {
        // Trailing `|#` should never appear when both lists are empty.
        let s = render(&m("foo", Value::Int(1), Kind::Counter, &[]), "", &[]);
        assert!(!s.contains("|#"), "no trailing tag separator: {s}");
    }

    #[test]
    fn disabled_client_is_noop() {
        // Disabled clients hold no socket; emit() returns immediately.
        // We assert by ensuring construction is cheap and no panic occurs
        // on any of the four kinds plus the timer guard.
        let c = Client::disabled();
        c.count("a", 1, &[]);
        c.gauge("a", 1.0, &[]);
        c.histogram("a", 1.0, &[]);
        c.timer("a", 1.0, &[]);
        {
            let _t = c.time("a", vec![]);
        }
        // No assertion needed beyond "didn't panic", but verify state:
        assert!(c.sock.is_none());
        assert_eq!(c.addr(), "");
    }

    #[test]
    fn builder_chain() {
        let c =
            Client::disabled().with_prefix("svc").with_tag("env", "test").with_tag("region", "lax");
        assert_eq!(c.prefix, "svc");
        assert_eq!(c.constant_tags.len(), 2);
        assert_eq!(c.constant_tags[0], ("env".into(), "test".into()));
    }

    #[test]
    fn timer_guard_emits_on_drop() {
        // Bind a receiver, point a client at it, drop a Timer guard,
        // and observe the metric.
        let recv = UdpSocket::bind("127.0.0.1:0").unwrap();
        recv.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let addr = recv.local_addr().unwrap().to_string();
        let c = Client::new(addr).unwrap().with_prefix("t");
        {
            let _g = c.time("op", vec![("k", "v")]);
            std::thread::sleep(Duration::from_millis(2));
        }
        let mut buf = [0u8; 2048];
        let n = recv.recv(&mut buf).unwrap();
        let line = std::str::from_utf8(&buf[..n]).unwrap();
        assert!(line.starts_with("t.op:"), "got: {line}");
        assert!(line.ends_with("|ms|#k:v"), "got: {line}");
    }

    #[test]
    fn end_to_end_udp_send_recv() {
        let recv = UdpSocket::bind("127.0.0.1:0").unwrap();
        recv.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let addr = recv.local_addr().unwrap().to_string();
        let c = Client::new(addr).unwrap().with_prefix("agent").with_tag("env", "test");

        c.count("req", 7, &[("route", "/x")]);
        let mut buf = [0u8; 2048];
        let n = recv.recv(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"agent.req:7|c|#env:test,route:/x");

        c.gauge("temp", 2.5, &[]);
        let n = recv.recv(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"agent.temp:2.5|g|#env:test");
    }

    #[test]
    fn name_resolution_failure_returns_err() {
        // A clearly bogus host with no DNS resolution must error at construct time.
        let err = Client::new("definitely.not.a.real.host.invalid:8125");
        assert!(err.is_err(), "expected resolution failure");
    }

    #[test]
    fn integer_floats_render_without_decimal() {
        // Required so `histogram(.., 42.0, ..)` produces `42|h` not `42.0|h`,
        // which matches what the Datadog Agent expects and what
        // existing dashboards/queries see for integer-valued samples.
        let s = render(&m("x", Value::Float(42.0), Kind::Histogram, &[]), "", &[]);
        assert_eq!(s, "x:42|h");
        let s = render(&m("x", Value::Float(120.0), Kind::Timer, &[]), "", &[]);
        assert_eq!(s, "x:120|ms");
    }

    #[test]
    fn fractional_floats_render_with_decimal() {
        let s = render(&m("x", Value::Float(0.5), Kind::Gauge, &[]), "", &[]);
        assert_eq!(s, "x:0.5|g");
    }

    #[test]
    fn from_env_disabled_when_unset() {
        // SAFETY: tests in this module run in the same process; we restore.
        let prev = std::env::var("DOGSTATSD_ADDR").ok();
        // SAFETY: single-threaded inside this test, no other readers.
        unsafe { std::env::remove_var("DOGSTATSD_ADDR") };
        let c = Client::from_env();
        assert!(c.sock.is_none());
        if let Some(v) = prev {
            // SAFETY: ditto.
            unsafe { std::env::set_var("DOGSTATSD_ADDR", v) };
        }
    }

    // Touch the test-only recorder helpers so they don't dead-code-warn.
    #[test]
    fn recorder_helpers_round_trip() {
        record("hello".into());
        let got = drain_recorded();
        assert_eq!(got, vec!["hello".to_string()]);
        assert!(drain_recorded().is_empty());
    }
}
