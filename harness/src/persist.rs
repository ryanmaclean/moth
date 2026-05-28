//! Persistence hook for `Session`. The harness defines the trait; concrete
//! impls live in the `persist` crate (file-backed) or whichever store a
//! caller wires in (SQLite, Redis, Durable Object, ...).

use crate::model::ChatMessage;

#[derive(Debug, Clone)]
pub struct StoreError(pub String);

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StoreError {}

/// Backs a `Session`'s message history.
///
/// `load` returns `Ok(None)` for an unseen id; `Ok(Some(_))` for a resumed
/// session. `append` is called after every successful turn with only the
/// messages that haven't been persisted yet (best-effort — an append error
/// logs and continues rather than failing the prompt). `snapshot` is an
/// optional best-effort compaction hook: stores that maintain an
/// append-only log can use it to fold the log into a snapshot.
pub trait SessionStore: Send + Sync + 'static {
    fn load(&self, key: &str) -> Result<Option<Vec<ChatMessage>>, StoreError>;
    /// Append `new_messages` to the end of the persisted history.
    fn append(&self, key: &str, new_messages: &[ChatMessage]) -> Result<(), StoreError>;
    /// Best-effort log compaction. The caller passes the canonical
    /// post-compaction history so a store that keeps an append log can
    /// fold it into a fresh snapshot. Default: no-op.
    fn snapshot(&self, _key: &str, _history: &[ChatMessage]) -> Result<(), StoreError> {
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod test_store {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    pub struct InMemoryStore {
        pub data: Mutex<HashMap<String, Vec<ChatMessage>>>,
        pub append_calls: Mutex<u32>,
    }

    impl InMemoryStore {
        pub fn new() -> Self {
            Self { data: Mutex::new(HashMap::new()), append_calls: Mutex::new(0) }
        }
    }

    impl SessionStore for InMemoryStore {
        fn load(&self, key: &str) -> Result<Option<Vec<ChatMessage>>, StoreError> {
            Ok(self.data.lock().unwrap().get(key).cloned())
        }
        fn append(&self, key: &str, new_messages: &[ChatMessage]) -> Result<(), StoreError> {
            *self.append_calls.lock().unwrap() += 1;
            let mut data = self.data.lock().unwrap();
            data.entry(key.to_string()).or_default().extend_from_slice(new_messages);
            Ok(())
        }
    }
}
