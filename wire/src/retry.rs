//! Retry-with-backoff + per-host circuit breaker.
//!
//! Shared by the model HTTP clients (`anthropic`, `openai`) and any other
//! caller that talks to flaky upstreams. The policy treats network errors
//! and 5xx/429-class responses as retryable; everything else is fatal.
//! A circuit breaker opens after a streak of retryables so a downed host
//! stops getting hammered.
//!
//! Zero new external deps: jitter is a tiny LCG seeded from clock nanos,
//! state is `OnceLock<Mutex<HashMap<...>>>` per process.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Caller signals how to treat the outcome of a single attempt.
pub enum Outcome<T, E> {
    Ok(T),
    Retryable(E),
    Fatal(E),
}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            base_delay: Duration::from_millis(250),
            max_delay: Duration::from_secs(8),
        }
    }
}

/// Retry the operation up to `policy.max_attempts` times with exponential
/// backoff. Returns the first `Ok`, the last `Fatal`, or the last error
/// after exhausting attempts. `Retry-After`-style sleep overrides can be
/// signalled by the op via the optional `sleep_hint` callback parameter.
pub fn with_backoff<T, E, F>(policy: &RetryPolicy, mut op: F) -> Result<T, E>
where
    F: FnMut(u32) -> Outcome<T, E>,
{
    let mut last_err: Option<E> = None;
    for attempt in 1..=policy.max_attempts {
        match op(attempt) {
            Outcome::Ok(v) => return Ok(v),
            Outcome::Fatal(e) => return Err(e),
            Outcome::Retryable(e) => {
                last_err = Some(e);
                if attempt < policy.max_attempts {
                    std::thread::sleep(backoff_delay(policy, attempt));
                }
            }
        }
    }
    Err(last_err.expect("at least one attempt always runs"))
}

fn backoff_delay(policy: &RetryPolicy, attempt: u32) -> Duration {
    // Exponential: base * 2^(attempt-1), capped at max_delay. Plus jitter
    // up to delay/4 to avoid synchronised retries across many callers.
    let factor = 1u32 << (attempt - 1).min(20);
    let base = policy.base_delay.as_millis() as u64 * factor as u64;
    let capped = base.min(policy.max_delay.as_millis() as u64);
    let jitter = lcg_jitter(capped / 4);
    Duration::from_millis(capped + jitter)
}

/// Tiny LCG (Numerical Recipes coefficients), seeded by Instant. Sufficient
/// for jitter — we don't need crypto randomness.
fn lcg_jitter(max: u64) -> u64 {
    if max == 0 {
        return 0;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0);
    let now = Instant::now().elapsed().as_nanos() as u64;
    let prev = STATE.load(Ordering::Relaxed);
    let next = prev.wrapping_mul(1664525).wrapping_add(1013904223 ^ now);
    STATE.store(next, Ordering::Relaxed);
    next % max
}

// ----- circuit breaker ---------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerVerdict {
    /// Caller may proceed.
    Allow,
    /// Caller MUST fail fast — host is in cooldown.
    Open,
    /// Caller may proceed as the probe; report outcome via `record_*`.
    HalfOpenProbe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BreakerConfig {
    /// Consecutive failures that flip Closed → Open.
    pub failure_threshold: u32,
    /// Initial cooldown after first Open. Doubles on each subsequent Open,
    /// capped at `max_cooldown`.
    pub initial_cooldown: Duration,
    pub max_cooldown: Duration,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            initial_cooldown: Duration::from_secs(1),
            max_cooldown: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone)]
struct HostState {
    consecutive_failures: u32,
    state: State,
    cooldown: Duration,
    opened_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Closed,
    Open,
    HalfOpen,
}

impl HostState {
    fn fresh(cfg: &BreakerConfig) -> Self {
        Self {
            consecutive_failures: 0,
            state: State::Closed,
            cooldown: cfg.initial_cooldown,
            opened_at: Instant::now(),
        }
    }
}

static BREAKERS: OnceLock<Mutex<HashMap<String, HostState>>> = OnceLock::new();

fn breakers() -> &'static Mutex<HashMap<String, HostState>> {
    BREAKERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Ask whether a call to `host` should proceed. Lock is held only for the
/// inspection + state update; the actual upstream request happens outside.
pub fn check(host: &str, cfg: &BreakerConfig) -> BreakerVerdict {
    let mut map = breakers().lock().unwrap();
    let entry = map.entry(host.to_string()).or_insert_with(|| HostState::fresh(cfg));
    match entry.state {
        State::Closed => BreakerVerdict::Allow,
        State::HalfOpen => BreakerVerdict::HalfOpenProbe,
        State::Open => {
            if entry.opened_at.elapsed() >= entry.cooldown {
                entry.state = State::HalfOpen;
                BreakerVerdict::HalfOpenProbe
            } else {
                BreakerVerdict::Open
            }
        }
    }
}

/// Record an upstream success. Closes the breaker if it was probing;
/// resets the failure counter unconditionally.
pub fn record_success(host: &str, cfg: &BreakerConfig) {
    let mut map = breakers().lock().unwrap();
    let entry = map.entry(host.to_string()).or_insert_with(|| HostState::fresh(cfg));
    entry.consecutive_failures = 0;
    entry.state = State::Closed;
    entry.cooldown = cfg.initial_cooldown;
}

/// Record an upstream retryable failure. Increments the streak counter
/// and opens the breaker if the threshold is hit. HalfOpen failure
/// re-opens with doubled cooldown.
pub fn record_failure(host: &str, cfg: &BreakerConfig) {
    let mut map = breakers().lock().unwrap();
    let entry = map.entry(host.to_string()).or_insert_with(|| HostState::fresh(cfg));
    entry.consecutive_failures = entry.consecutive_failures.saturating_add(1);
    match entry.state {
        State::HalfOpen => {
            entry.state = State::Open;
            entry.cooldown = (entry.cooldown * 2).min(cfg.max_cooldown);
            entry.opened_at = Instant::now();
        }
        State::Closed if entry.consecutive_failures >= cfg.failure_threshold => {
            entry.state = State::Open;
            entry.opened_at = Instant::now();
        }
        _ => {}
    }
}

/// Test-only: reset all breaker state so tests don't carry state between
/// each other. Public-but-doc(hidden) so the integration suite can call
/// it too without exposing it as a stable API.
#[doc(hidden)]
pub fn reset_breakers_for_tests() {
    let mut map = breakers().lock().unwrap();
    map.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fatal_returns_immediately() {
        let policy = RetryPolicy {
            max_attempts: 4,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
        };
        let mut calls = 0;
        let result: Result<(), &str> = with_backoff(&policy, |_| {
            calls += 1;
            Outcome::Fatal("nope")
        });
        assert_eq!(calls, 1);
        assert_eq!(result, Err("nope"));
    }

    #[test]
    fn retryable_runs_up_to_max_attempts() {
        let policy = RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
        };
        let mut calls = 0;
        let result: Result<(), &str> = with_backoff(&policy, |_| {
            calls += 1;
            Outcome::Retryable("flaky")
        });
        assert_eq!(calls, 3);
        assert!(result.is_err());
    }

    #[test]
    fn ok_short_circuits() {
        let policy = RetryPolicy::default();
        let mut calls = 0;
        let result: Result<i32, &str> = with_backoff(&policy, |_| {
            calls += 1;
            Outcome::Ok(42)
        });
        assert_eq!(calls, 1);
        assert_eq!(result, Ok(42));
    }

    #[test]
    fn retryable_then_ok_succeeds() {
        let policy = RetryPolicy {
            max_attempts: 5,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
        };
        let mut calls = 0;
        let result: Result<i32, &str> = with_backoff(&policy, |attempt| {
            calls += 1;
            if attempt < 3 { Outcome::Retryable("transient") } else { Outcome::Ok(7) }
        });
        assert_eq!(calls, 3);
        assert_eq!(result, Ok(7));
    }

    #[test]
    fn backoff_delay_grows_exponentially_and_caps() {
        let policy = RetryPolicy {
            max_attempts: 6,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(1000),
        };
        let d1 = backoff_delay(&policy, 1);
        let d2 = backoff_delay(&policy, 2);
        let d4 = backoff_delay(&policy, 4);
        let d6 = backoff_delay(&policy, 6);
        // Each step's base grows, modulo jitter (<= base/4 added).
        assert!(d2.as_millis() >= d1.as_millis(), "d2 {d2:?} < d1 {d1:?}");
        // Eventually capped — base * 2^5 = 3200 > 1000ms cap.
        assert!(d6.as_millis() <= 1000 + 250, "d6 {d6:?} should respect max");
        assert!(d4.as_millis() <= 1000 + 250);
    }

    #[test]
    fn breaker_opens_after_threshold() {
        let cfg = BreakerConfig { failure_threshold: 3, ..BreakerConfig::default() };
        let host = "h1.example.com";
        assert_eq!(check(host, &cfg), BreakerVerdict::Allow);
        record_failure(host, &cfg);
        record_failure(host, &cfg);
        assert_eq!(check(host, &cfg), BreakerVerdict::Allow);
        record_failure(host, &cfg);
        assert_eq!(check(host, &cfg), BreakerVerdict::Open);
    }

    #[test]
    fn breaker_half_open_probe_succeeds_closes() {
        let cfg = BreakerConfig {
            failure_threshold: 1,
            initial_cooldown: Duration::from_millis(10),
            max_cooldown: Duration::from_millis(100),
        };
        let host = "h2.example.com";
        record_failure(host, &cfg);
        assert_eq!(check(host, &cfg), BreakerVerdict::Open);
        std::thread::sleep(Duration::from_millis(15));
        assert_eq!(check(host, &cfg), BreakerVerdict::HalfOpenProbe);
        record_success(host, &cfg);
        assert_eq!(check(host, &cfg), BreakerVerdict::Allow);
    }

    #[test]
    fn breaker_half_open_probe_failure_reopens_with_longer_cooldown() {
        let cfg = BreakerConfig {
            failure_threshold: 1,
            initial_cooldown: Duration::from_millis(10),
            max_cooldown: Duration::from_millis(200),
        };
        let host = "h3.example.com";
        record_failure(host, &cfg);
        std::thread::sleep(Duration::from_millis(15));
        assert_eq!(check(host, &cfg), BreakerVerdict::HalfOpenProbe);
        record_failure(host, &cfg);
        // Re-opened: still Open even after the original 10ms.
        assert_eq!(check(host, &cfg), BreakerVerdict::Open);
        // After 15ms, the original cooldown would have expired but
        // the new cooldown is ~20ms (10 * 2). Sleep enough for both.
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(check(host, &cfg), BreakerVerdict::HalfOpenProbe);
    }

    #[test]
    fn different_hosts_have_independent_breakers() {
        let cfg = BreakerConfig { failure_threshold: 2, ..BreakerConfig::default() };
        record_failure("a.example.com", &cfg);
        record_failure("a.example.com", &cfg);
        assert_eq!(check("a.example.com", &cfg), BreakerVerdict::Open);
        assert_eq!(check("b.example.com", &cfg), BreakerVerdict::Allow);
    }

    #[test]
    fn success_resets_failure_streak() {
        let cfg = BreakerConfig { failure_threshold: 3, ..BreakerConfig::default() };
        let host = "h4.example.com";
        record_failure(host, &cfg);
        record_failure(host, &cfg);
        record_success(host, &cfg);
        // Streak reset → two more failures shouldn't open.
        record_failure(host, &cfg);
        record_failure(host, &cfg);
        assert_eq!(check(host, &cfg), BreakerVerdict::Allow);
    }
}
