# syntax=docker/dockerfile:1.6
#
# Multi-stage build for the `agent` CLI.
#   Stage 1 (builder): rust + native + aarch64 cross toolchain. Reusable
#                      by CI so PR/release builds run in the *same*
#                      environment as `docker build .` on a developer
#                      laptop. No source copy — purely an environment.
#   Stage 2 (build):   FROM builder, copies the workspace and runs
#                      cargo build offline. Strips the binary.
#   Stage 3 (runtime): distroless, only the static binary.
# Final image is ~48 MB.

FROM rust:1.94.1-slim-bookworm AS builder

# apt deps:
#   build-essential pkg-config perl make — vendored C sources
#       (zlib, openssl, libcurl via curl-sys) need these to build.
#   git — the `git` crate (git/src/lib.rs) shells out to `git(1)` via
#       Command::new("git") for branch/worktree/status. Without it,
#       every test in git/src/lib.rs fails Io(NotFound) at `git init`.
#       rust:slim-bookworm does not bundle git.
#   gcc-aarch64-linux-gnu — aarch64 cross *compiler*.
#   libc6-dev-arm64-cross linux-libc-dev-arm64-cross — aarch64
#       *sysroot* (glibc + kernel headers). Without them, C build
#       scripts in libz-sys / openssl-sys / curl-sys fail
#       'zlib.h: No such file or directory' when cross-compiling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential pkg-config perl make \
        git \
        gcc-aarch64-linux-gnu \
        libc6-dev-arm64-cross \
        linux-libc-dev-arm64-cross \
    && rm -rf /var/lib/apt/lists/*

# rust-toolchain.toml at the workspace root pins channel + components +
# targets, but `rustup target add` here means a `cargo` invocation that
# doesn't see rust-toolchain.toml (e.g. -C/--manifest-path from outside
# the tree) still has the aarch64 std available.
RUN rustup target add aarch64-unknown-linux-gnu

# Cross-compile env defaults so `cargo ... --target aarch64-unknown-linux-gnu`
# Just Works without per-step env in CI.
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc \
    AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar

WORKDIR /src

FROM builder AS build

COPY . .

RUN cargo build --release --locked --offline --frozen -p cli \
    && strip target/release/agent

FROM gcr.io/distroless/cc-debian12 AS runtime

# Parameterized so the label tracks whatever owner/repo the image was
# built in. Release CI passes --build-arg REPO_SOURCE=https://github.com/${{ github.repository }}.
ARG REPO_SOURCE="https://github.com/ryanmaclean/moth"
LABEL org.opencontainers.image.source="${REPO_SOURCE}"
LABEL org.opencontainers.image.description="Minimal Rust agent harness; vendored deps, libcurl + OpenSSL via curl-sys."
LABEL org.opencontainers.image.licenses="MIT"

COPY --from=build /src/target/release/agent /usr/local/bin/agent

ENTRYPOINT ["/usr/local/bin/agent"]
CMD ["--help"]
