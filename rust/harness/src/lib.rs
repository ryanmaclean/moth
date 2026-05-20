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
mod persist;
mod sandbox;
mod session;
mod tools;

pub use adapters::{AnthropicModel, AuditedShell, OpenAiModel};
pub use persist::{SessionStore, StoreError};
pub use instance::{Instance, InstanceMsg};
pub use model::{
    ChatMessage, ContentBlock, MockModel, Model, ModelError, ModelEvent, ModelRequest,
    Role, ToolDef,
};
pub use sandbox::{MockSandbox, Sandbox, SandboxError, ShellResult};
pub use session::{HarnessState, PromptResult, Session, SessionError, SessionMsg};
pub use tools::{BashTool, Tool, ToolCtx, ToolError, default_tools};
