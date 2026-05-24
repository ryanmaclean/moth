//! Transports for the MCP client.
//!
//! Two implementations live behind the [`Transport`] trait:
//!
//! - [`StdioTransport`]: spawn a child MCP server and exchange newline-
//!   delimited JSON over its stdin/stdout. Used by every local MCP
//!   package.
//! - [`HttpTransport`]: streamable-HTTP transport per the MCP spec.
//!   Single POST endpoint; the server replies with either a JSON body
//!   or an SSE stream of `data:` frames carrying JSON-RPC messages.
//!
//! The trait is line-oriented so the JSON-RPC layer stays transport-
//! agnostic; the HTTP impl buffers responses and drains them line-by-line
//! to match.

mod http;
mod stdio;

/// Abstract send/recv pair used by the MCP client. Errors are plain
/// strings — transports report enough context inline that a typed enum
/// would be overkill.
pub trait Transport: Send + Sync {
    /// Send a single JSON-RPC frame. Implementations append a trailing
    /// `\n` (where the framing requires it); callers pass the payload
    /// without one.
    fn send_line(&self, line: &[u8]) -> Result<(), String>;

    /// Block until the next frame is available. Returns the payload
    /// without the trailing `\n`. EOF/closed channel becomes an error.
    fn recv_line(&self) -> Result<Vec<u8>, String>;
}

pub use http::HttpTransport;
pub use stdio::StdioTransport;
