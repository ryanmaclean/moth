//! Instance actor: owns the sandbox; serialises shell access.

use std::sync::mpsc::SyncSender;

use actor::Actor;

use crate::sandbox::{Sandbox, SandboxError, ShellResult};

pub struct Instance {
    pub id: String,
    pub sandbox: Box<dyn Sandbox>,
}

impl Instance {
    pub fn new(id: impl Into<String>, sandbox: Box<dyn Sandbox>) -> Self {
        Self { id: id.into(), sandbox }
    }
}

pub enum InstanceMsg {
    Shell {
        cmd: String,
        reply: SyncSender<Result<ShellResult, SandboxError>>,
    },
}

impl Actor for Instance {
    type Msg = InstanceMsg;

    fn handle(&mut self, msg: InstanceMsg) {
        match msg {
            InstanceMsg::Shell { cmd, reply } => {
                let _ = reply.send(self.sandbox.execute(&cmd));
            }
        }
    }
}
