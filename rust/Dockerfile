# syntax=docker/dockerfile:1.6
#
# Multi-stage build for the `agent` CLI.
#   Stage 1: compile from vendored sources (offline). Strips the binary.
#   Stage 2: distroless runtime with only the static binary.
# Final image is ~48 MB.

FROM rust:1.90-slim-bookworm AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential pkg-config perl make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cargo build --release --locked --offline --frozen -p cli \
    && strip target/release/agent

FROM gcr.io/distroless/cc-debian12 AS runtime

LABEL org.opencontainers.image.source="https://github.com/mattpocock/sandcastle"
LABEL org.opencontainers.image.description="Minimal Rust agent harness; vendored deps, libcurl + OpenSSL via curl-sys."
LABEL org.opencontainers.image.licenses="MIT"

COPY --from=build /src/target/release/agent /usr/local/bin/agent

ENTRYPOINT ["/usr/local/bin/agent"]
CMD ["--help"]
