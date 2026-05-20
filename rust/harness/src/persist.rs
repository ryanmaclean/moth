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

/// Backs a `Session`'s message history. `load` returns `Ok(None)` for an
/// unseen id; `Ok(Some(_))` for a resumed session. `save` is called after
/// every successful turn (best-effort — a save error logs and continues
/// rather than failing the prompt).
pub trait SessionStore: Send + Sync + 'static {
    fn load(&self, key: &str) -> Result<Option<Vec<ChatMessage>>, StoreError>;
    fn save(&self, key: &str, history: &[ChatMessage]) -> Result<(), StoreError>;
}

#[cfg(test)]
pub(crate) mod test_store {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    pub struct InMemoryStore {
        pub data: Mutex<HashMap<String, Vec<ChatMessage>>>,
        pub save_calls: Mutex<u32>,
    }

    impl InMemoryStore {
        pub fn new() -> Self {
            Self { data: Mutex::new(HashMap::new()), save_calls: Mutex::new(0) }
        }
    }

    impl SessionStore for InMemoryStore {
        fn load(&self, key: &str) -> Result<Option<Vec<ChatMessage>>, StoreError> {
            Ok(self.data.lock().unwrap().get(key).cloned())
        }
        fn save(&self, key: &str, history: &[ChatMessage]) -> Result<(), StoreError> {
            *self.save_calls.lock().unwrap() += 1;
            self.data.lock().unwrap().insert(key.to_string(), history.to_vec());
            Ok(())
        }
    }
}
