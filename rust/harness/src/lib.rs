//! Orchestration layer for the Rust agent harness.
//!
//! Two actors — `Instance` (owns the sandbox) and `Session` (owns message
//! history + drives the iteration loop) — plus the `Model` and `Sandbox`
//! traits they consume. `Harness` is a passive value (`HarnessState`) held
//! by each `Session`: it's only model defaults + an `ActorRef` to the
//! instance, so promoting it to its own actor would add hops without
//! serialising any state.
//!
//! See Flue's instance/harness/session hierarchy for the conceptual model;
//! the implementation collapses Harness into a value type.

mod adapters;
mod instance;
mod model;
mod sandbox;
mod session;

pub use adapters::{AnthropicModel, AuditedShell};
pub use instance::{Instance, InstanceMsg};
pub use model::{
    ChatMessage, ContentBlock, MockModel, Model, ModelError, ModelEvent, ModelRequest,
    Role, ToolDef,
};
pub use sandbox::{MockSandbox, Sandbox, SandboxError, ShellResult};
pub use session::{
    bash_tool, HarnessState, PromptResult, Session, SessionError, SessionMsg,
};
