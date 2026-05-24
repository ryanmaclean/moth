//! vshell parse + dispatch benches.
//!
//! `true` is a built-in — measures pure parse + dispatch with no spawn.
//! `echo $X` exercises variable expansion plus the `echo` built-in.

use std::hint::black_box;

use vshell::VShell;

use crate::bench_helper::bench;

#[test]
fn bench_vshell_true() {
    bench("vshell::execute(\"true\")", || {
        // Each iteration creates a fresh shell to measure full fixed
        // overhead per invocation; cwd() pulls in std::env::current_dir
        // which is a syscall. If that dominates we'll see it.
        let mut sh = VShell::new();
        let r = black_box(sh.execute(black_box("true")));
        debug_assert_eq!(r.exit_code, 0);
        drop(black_box(r));
    });
}

#[test]
fn bench_vshell_echo_var() {
    // Build the shell once; we want to measure `execute` itself, not
    // VShell::new (which calls current_dir). The previous test does
    // measure new+execute together.
    let mut sh = VShell::new();
    sh.set_env("X", "hello");
    bench("vshell::execute(\"echo $X\") (X exported)", || {
        let r = black_box(sh.execute(black_box("echo $X")));
        debug_assert_eq!(r.exit_code, 0);
        drop(black_box(r));
    });
}
