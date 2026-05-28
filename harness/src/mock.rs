//! Scripted [`Model`] implementation for tests, demos, and CI runs.
//!
//! A `MockModel` is constructed with a list of scripted "turns"; each turn
//! is the sequence of [`ModelEvent`]s the mock will yield on a single call
//! to [`Model::stream`]. Every call to `stream` advances an internal cursor
//! one turn forward, **cycling back to turn 0 once the script list is
//! exhausted**. This lets a single short script drive arbitrarily many
//! turns — useful for the CLI `--mock` flag (canned response replayed each
//! call) and for soak tests that don't want to pre-allocate N copies of the
//! same events.
//!
//! Every incoming [`ModelRequest`] is appended to `seen` (under a `Mutex`)
//! so tests can assert on what the harness sent to the model.
//!
//! ```
//! use harness::{MockModel, Model, ModelEvent, ModelRequest};
//!
//! let m = MockModel::single(vec![
//!     ModelEvent::TextDelta("hi".into()),
//!     ModelEvent::Stop { reason: Some("end_turn".into()) },
//! ]);
//! let req = ModelRequest {
//!     system: None,
//!     messages: vec![],
//!     max_tokens: 16,
//!     tools: vec![],
//! };
//! let events: Vec<_> = m.stream(req).collect();
//! assert_eq!(events.len(), 2);
//! ```

use std::sync::Mutex;

use crate::model::{Model, ModelError, ModelEvent, ModelRequest};

/// Scripted, no-network [`Model`] for tests, CLI `--mock`, and demos.
///
/// Holds a list of scripted turns (`Vec<Vec<ModelEvent>>`). Each call to
/// [`Model::stream`] yields the next turn's events; when the cursor passes
/// the last turn it wraps back to turn 0. Empty script lists yield empty
/// iterators forever.
///
/// `seen` records every [`ModelRequest`] the harness has sent to the
/// model, in order, so tests can inspect prompt/tool wiring.
pub struct MockModel {
    /// Script of turns. Never mutated after construction; the cursor below
    /// is what advances. (Kept under `Mutex` to satisfy `Send + Sync` and
    /// future-proof against test helpers that want to swap scripts in.)
    scripts: Mutex<Vec<Vec<ModelEvent>>>,
    /// Next-turn index. Wraps modulo `scripts.len()` so an exhausted
    /// script cycles instead of returning empty.
    cursor: Mutex<usize>,
    /// Every `ModelRequest` we've handled, in arrival order.
    pub seen: Mutex<Vec<ModelRequest>>,
}

impl MockModel {
    /// Build a mock with `scripts.len()` distinct turns. The first call to
    /// [`Model::stream`] replays `scripts[0]`, the second replays
    /// `scripts[1]`, and so on; once exhausted the cursor wraps to 0.
    pub fn new(scripts: Vec<Vec<ModelEvent>>) -> Self {
        Self { scripts: Mutex::new(scripts), cursor: Mutex::new(0), seen: Mutex::new(Vec::new()) }
    }

    /// Convenience for the common single-turn case (e.g. one `TextDelta`
    /// + one `Stop`). The same events are replayed on every call.
    pub fn single(events: Vec<ModelEvent>) -> Self {
        Self::new(vec![events])
    }
}

impl Model for MockModel {
    fn stream(
        &self,
        req: ModelRequest,
    ) -> Box<dyn Iterator<Item = Result<ModelEvent, ModelError>> + Send> {
        self.seen.lock().unwrap().push(req);
        let scripts = self.scripts.lock().unwrap();
        if scripts.is_empty() {
            return Box::new(std::iter::empty());
        }
        let mut cursor = self.cursor.lock().unwrap();
        let idx = *cursor % scripts.len();
        *cursor = cursor.wrapping_add(1);
        let next = scripts[idx].clone();
        Box::new(next.into_iter().map(Ok))
    }
}
